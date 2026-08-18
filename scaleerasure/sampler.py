"""ScaleErasure sampling on top of the unmodified Infinity backbone."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from infinity.models.basic import CrossAttnBlock
from infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_

Condition = tuple[torch.Tensor, list[int], torch.Tensor, int]


@dataclass(frozen=True)
class TokenMaskSnapshot:
    """One image-token mask captured at an intervention scale.

    ``relevance`` and ``binary_mask`` are stored as CPU tensors so an optional
    debug trace does not retain the model's computation graph or GPU memory.
    The flattened order is the same ``t, h, w`` order used by Infinity's
    autoregressive logits.  The exporter can therefore reconstruct either the
    complete token volume or the first-frame 2-D view used by the legacy code.
    """

    scale_index: int
    scale_shape: tuple[int, int, int]
    relevance: torch.Tensor
    binary_mask: torch.Tensor
    threshold: float
    selected_layers: tuple[int, ...]


def _schedule(value: float | Sequence[float], length: int) -> list[float]:
    if isinstance(value, (list, tuple)):
        schedule = [float(item) for item in value]
    else:
        schedule = [float(value)] * length
    if len(schedule) < length:
        raise ValueError(f"schedule has {len(schedule)} values, expected {length}")
    return schedule


def _condition_lengths(condition: Condition) -> list[int]:
    return [int(length) for length in condition[1]]


def _concat_conditions(conditions: Sequence[Condition]) -> Condition:
    compact_parts = [condition[0] for condition in conditions]
    lengths = [length for condition in conditions for length in _condition_lengths(condition)]
    compact = torch.cat(compact_parts, dim=0)
    cumulative_lengths = np.cumsum([0, *lengths], dtype=np.int64)
    cu_seqlens = torch.tensor(
        cumulative_lengths,
        dtype=torch.int32,
        device=compact.device,
    )
    return compact, lengths, cu_seqlens, max(lengths)


def _build_unconditional_condition(
    model,
    prompt_condition: Condition,
    negative_condition: Condition | None,
) -> Condition:
    if negative_condition is not None:
        return negative_condition

    prompt_compact, prompt_lengths, prompt_cu_seqlens, prompt_max_length = prompt_condition
    unconditional_compact = prompt_compact.clone()
    cursor = 0
    unconditional_tokens = model.cfg_uncond.to(
        device=prompt_compact.device,
        dtype=prompt_compact.dtype,
    )
    for length in prompt_lengths:
        unconditional_compact[cursor : cursor + length] = unconditional_tokens[:length]
        cursor += length
    return (
        unconditional_compact,
        list(prompt_lengths),
        prompt_cu_seqlens.clone(),
        prompt_max_length,
    )


def _project_conditions(
    model,
    condition: Condition,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
    compact, lengths, cu_seqlens, max_length = condition
    normalized = model.text_norm(compact)
    pooled = model.text_proj_for_sos((normalized, cu_seqlens, max_length))
    projected = model.text_proj_for_ca(normalized)
    return pooled, (projected, cu_seqlens, max_length)


def _collect_token_relevance(
    block,
    reference_concept_embeddings: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    if not isinstance(block, CrossAttnBlock):
        return None
    if block.ca.last_query_heads is None:
        return None

    concept_scores = []
    for concept_embedding in reference_concept_embeddings:
        # The old implementation calls compute_external_attention(q=None),
        # which scores only the first conditional sample even when B > 1.
        scores = block.ca.score_reference_embedding(concept_embedding, batch_size=1)
        concept_scores.append(scores)
    if not concept_scores:
        return None

    stacked_scores = torch.stack(concept_scores, dim=0)
    # Match the original pixel-wise competition: normalize across concepts
    # independently for each attention head, then average the heads.
    return torch.softmax(stacked_scores, dim=0)[0].mean(dim=-1)


def _set_query_capture(model, enabled: bool) -> list:
    attention_modules = []
    for block in model.unregistered_blocks:
        if isinstance(block, CrossAttnBlock):
            block.ca.set_query_capture(enabled)
            attention_modules.append(block.ca)
    return attention_modules


def _clear_kv_cache(model) -> None:
    for block in model.unregistered_blocks:
        attention = block.sa if isinstance(block, CrossAttnBlock) else block.attn
        attention.kv_caching(False)


def _set_kv_cache(model) -> None:
    for block in model.unregistered_blocks:
        attention = block.sa if isinstance(block, CrossAttnBlock) else block.attn
        attention.kv_caching(True)


@torch.no_grad()
def autoregressive_infer_scaleerasure(
    model,
    vae,
    scale_schedule: Sequence[tuple[int, int, int]],
    prompt_condition: Condition,
    safe_condition: Condition,
    unsafe_condition: Condition,
    reference_concept_embeddings: Sequence[torch.Tensor],
    *,
    seed: int | None,
    cfg_scale: float | Sequence[float] = 3.0,
    temperature: float | Sequence[float] = 0.5,
    token_mask_threshold: float = 0.332,
    bit_channel_threshold: float = 0.1,
    unsafe_penalty_weight: float = 3.0,
    safe_guidance_warmup_steps: int = 2,
    token_selection_layers: Sequence[int] = (1, 2, 3, 4, 5),
    token_selection_scales: Sequence[int] = tuple(range(13)),
    scale_start: int = 0,
    scale_end: int | None = 8,
    top_k: int = 1,
    top_p: float = 0.97,
    batch_size: int = 1,
    vae_type: int = 32,
    mask_snapshots: list[TokenMaskSnapshot] | None = None,
) -> torch.Tensor:
    """Generate one image with scale-, token-, and bit-selective guidance.

    When ``mask_snapshots`` is provided, append the token relevance and binary
    mask computed at every active intervention scale.  This is intentionally an
    opt-in diagnostic path and is not touched during normal generation.
    """

    scale_schedule = list(scale_schedule)
    num_scales = len(scale_schedule)
    cfg_schedule = _schedule(cfg_scale, num_scales)
    temperature_schedule = _schedule(temperature, num_scales)
    if any(value <= 0 for value in temperature_schedule):
        raise ValueError("temperature must be positive")
    if unsafe_penalty_weight < 0:
        raise ValueError("unsafe_penalty_weight must be non-negative")
    if safe_guidance_warmup_steps < 0:
        raise ValueError("safe_guidance_warmup_steps must be non-negative")
    if not reference_concept_embeddings:
        raise ValueError("at least one reference concept is required")

    selected_layers = {int(layer) for layer in token_selection_layers}
    selected_scales = {int(scale) for scale in token_selection_scales}
    prompt_unconditional = _build_unconditional_condition(model, prompt_condition, None)
    full_condition = _concat_conditions(
        [prompt_condition, prompt_unconditional, safe_condition, unsafe_condition]
    )
    cfg_condition = _concat_conditions([prompt_condition, prompt_unconditional])

    cond_BD, full_ca_kv = _project_conditions(model, full_condition)
    cond_BD_cfg, cfg_ca_kv = _project_conditions(model, cfg_condition)
    # Match Infinity's inference path: shared AdaLN is evaluated in fp32 even
    # when the surrounding transformer pass uses bf16 autocast.
    with torch.autocast("cuda", enabled=False):
        cond_BD_or_gss = model.shared_ada_lin(cond_BD.float()).float().contiguous()
        cond_BD_or_gss_cfg = model.shared_ada_lin(cond_BD_cfg.float()).float().contiguous()

    full_batch_size = 4 * batch_size
    current_batch_size = full_batch_size
    last_stage = cond_BD.unsqueeze(1).expand(current_batch_size, 1, -1) + model.pos_start.expand(
        current_batch_size, 1, -1
    )
    current_condition = full_ca_kv
    current_cond_BD = cond_BD
    current_cond_BD_or_gss = cond_BD_or_gss
    scale_end_reached = False
    guidance_step = 0
    # Preserve the legacy implementation's per-scale threshold state.  The
    # historical code updates this value inside the autoregressive loop, so
    # with tau=0.5 it evolves as 0.1 -> 0.2 -> 0.4 -> ... rather than
    # recomputing 0.1 / tau independently at every scale.
    legacy_fine_grained_threshold = float(bit_channel_threshold)

    if seed is None:
        rng = None
    else:
        model.rng.manual_seed(seed)
        rng = model.rng

    if model.apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule

    summed_codes = 0
    num_scales_minus_one = num_scales - 1
    query_modules = _set_query_capture(model, True)
    _set_kv_cache(model)

    for attention_module in query_modules:
        attention_module.last_query_heads = None

    try:
        for scale_index, scale_shape in enumerate(scale_schedule):
            if scale_end is not None and not scale_end_reached and scale_index >= scale_end:
                last_stage = last_stage[: 2 * batch_size]
                current_condition = cfg_ca_kv
                current_cond_BD = cond_BD_cfg
                current_cond_BD_or_gss = cond_BD_or_gss_cfg
                current_batch_size = 2 * batch_size
                for block in model.unregistered_blocks:
                    attention = block.sa if isinstance(block, CrossAttnBlock) else block.attn
                    attention.kv_cache_shrink(2 * batch_size)
                scale_end_reached = True

            if not scale_end_reached and scale_index >= scale_start:
                legacy_fine_grained_threshold /= temperature_schedule[scale_index]

            cfg_value = cfg_schedule[scale_index]
            need_to_pad = 0
            attention_fn = None
            if model.use_flex_attn:
                attention_fn = model.attn_fn_compile_dict.get(
                    tuple(scale_schedule[: scale_index + 1])
                )

            relevance_by_layer = []
            layer_index = 0
            block_groups = model.block_chunks if model.num_block_chunks > 1 else [model.blocks]
            for block_group_index, block_group in enumerate(block_groups):
                if model.add_lvl_embeding_only_first_block and block_group_index == 0:
                    last_stage = model.add_lvl_embeding(
                        last_stage,
                        scale_index,
                        scale_schedule,
                        need_to_pad=need_to_pad,
                    )
                if not model.add_lvl_embeding_only_first_block:
                    last_stage = model.add_lvl_embeding(
                        last_stage,
                        scale_index,
                        scale_schedule,
                        need_to_pad=need_to_pad,
                    )

                blocks = block_group.module if hasattr(block_group, "module") else block_group
                for block in blocks:
                    last_stage = block(
                        x=last_stage,
                        cond_BD=current_cond_BD_or_gss,
                        ca_kv=current_condition,
                        attn_bias_or_two_vector=None,
                        attn_fn=attention_fn,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=model.rope2d_freqs_grid,
                        scale_ind=scale_index,
                    )
                    if (
                        not scale_end_reached
                        and scale_index in selected_scales
                        and layer_index in selected_layers
                    ):
                        relevance = _collect_token_relevance(
                            block,
                            reference_concept_embeddings,
                        )
                        if relevance is not None:
                            relevance_by_layer.append(relevance)
                    layer_index += 1

            if cfg_value != 1:
                if scale_end_reached or scale_index < scale_start:
                    # Infinity's reference implementation only evaluates the
                    # prompt/unconditional branches before the intervention
                    # window. Besides doing less work, keeping the batch at
                    # 2B preserves its numerical path exactly.
                    logits = model.get_logits(
                        last_stage[: 2 * batch_size],
                        current_cond_BD[: 2 * batch_size],
                    ).mul(1 / temperature_schedule[scale_index])
                    prompt_logits = logits[:batch_size]
                    unconditional_logits = logits[batch_size : 2 * batch_size]
                    logits = cfg_value * prompt_logits + (1 - cfg_value) * unconditional_logits
                else:
                    if not relevance_by_layer:
                        raise RuntimeError(
                            "token relevance was not collected; check token_selection_layers"
                        )
                    logits = model.get_logits(last_stage, current_cond_BD).mul(
                        1 / temperature_schedule[scale_index]
                    )
                    prompt_logits = logits[:batch_size]
                    unconditional_logits = logits[batch_size : 2 * batch_size]
                    cfg_logits = cfg_value * prompt_logits + (1 - cfg_value) * unconditional_logits
                    token_relevance = torch.stack(relevance_by_layer).mean(dim=0)
                    token_mask = (token_relevance >= token_mask_threshold).to(cfg_logits.dtype)
                    if mask_snapshots is not None:
                        mask_snapshots.append(
                            TokenMaskSnapshot(
                                scale_index=scale_index,
                                scale_shape=tuple(int(value) for value in scale_shape),
                                relevance=token_relevance.detach().float().cpu(),
                                binary_mask=token_mask.squeeze(-1).detach().to(torch.uint8).cpu(),
                                threshold=float(token_mask_threshold),
                                selected_layers=tuple(sorted(selected_layers)),
                            )
                        )
                    token_mask = token_mask.unsqueeze(-1)
                    scaled_bit_channel_threshold = legacy_fine_grained_threshold

                    safe_logits = logits[2 * batch_size : 3 * batch_size]
                    unsafe_logits = logits[3 * batch_size : 4 * batch_size]
                    safe_guidance_logits = (
                        cfg_value * safe_logits
                        + (1 - cfg_value) * unconditional_logits
                        - unsafe_penalty_weight * (unsafe_logits - unconditional_logits)
                    )
                    bit_mask = (unsafe_logits - prompt_logits).abs() < scaled_bit_channel_threshold
                    selective_logits = torch.where(
                        bit_mask,
                        safe_guidance_logits,
                        cfg_logits,
                    )
                    if guidance_step < safe_guidance_warmup_steps:
                        logits = cfg_logits
                    else:
                        logits = (1 - token_mask) * cfg_logits + token_mask * selective_logits
                    guidance_step += 1
            else:
                logits = model.get_logits(
                    last_stage[:batch_size], current_cond_BD[:batch_size]
                ).mul(1 / temperature_schedule[scale_index])

            if model.use_bit_label:
                batch_logits, sequence_length = logits.shape[:2]
                bit_logits = logits.reshape(batch_logits, -1, 2)
                sampled_bits = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    bit_logits,
                    rng=rng,
                    top_k=top_k or model.top_k,
                    top_p=top_p or model.top_p,
                    num_samples=1,
                )[:, :, 0]
                sampled_bits = sampled_bits.reshape(batch_logits, sequence_length, -1)
                sampled_indices = sampled_bits
            else:
                sampled_indices = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits,
                    rng=rng,
                    top_k=top_k or model.top_k,
                    top_p=top_p or model.top_p,
                    num_samples=1,
                )[:, :, 0]

            if vae_type == 0:
                raise ValueError("ScaleErasure currently requires a bitwise VAE")
            assert scale_shape[0] == 1
            sampled_indices = sampled_indices[:batch_size]
            sampled_indices = sampled_indices.reshape(
                batch_size, scale_shape[1], scale_shape[2], -1
            )
            if model.apply_spatial_patchify:
                sampled_indices = sampled_indices.permute(0, 3, 1, 2)
                sampled_indices = F.pixel_shuffle(sampled_indices, 2)
                sampled_indices = sampled_indices.permute(0, 2, 3, 1)
            sampled_indices = sampled_indices.unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(
                sampled_indices,
                label_type="bit_label",
            )
            if scale_index != num_scales_minus_one:
                summed_codes = summed_codes + F.interpolate(
                    codes,
                    size=vae_scale_schedule[-1],
                    mode=vae.quantizer.z_interplote_up,
                )
                last_stage = F.interpolate(
                    summed_codes,
                    size=vae_scale_schedule[scale_index + 1],
                    mode=vae.quantizer.z_interplote_up,
                ).squeeze(-3)
                if model.apply_spatial_patchify:
                    last_stage = F.pixel_unshuffle(last_stage, 2)
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1).permute(0, 2, 1)
            else:
                summed_codes = summed_codes + codes

            if scale_index != num_scales_minus_one:
                last_stage = model.word_embed(model.norm0_ve(last_stage))
                last_stage = last_stage.repeat(current_batch_size // batch_size, 1, 1)

        image = vae.decode(summed_codes.squeeze(-3))
    finally:
        for attention_module in query_modules:
            attention_module.set_query_capture(False)
        _clear_kv_cache(model)

    image = (image + 1) / 2
    return image.permute(0, 2, 3, 1).mul(255).to(torch.uint8).flip(dims=(3,))[0]


__all__ = ["TokenMaskSnapshot", "autoregressive_infer_scaleerasure"]
