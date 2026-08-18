"""Infinity loading and prompt preparation for ScaleErasure inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer

from .sampler import TokenMaskSnapshot, autoregressive_infer_scaleerasure


def encode_prompt(text_tokenizer, text_encoder, prompt: str):
    """Encode a prompt into Infinity's compact text-KV representation."""

    tokens = text_tokenizer(
        text=[prompt],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    device = next(text_encoder.parameters()).device
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    attention_mask = tokens.attention_mask.to(device, non_blocking=True)

    with torch.no_grad():
        text_features = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"].float()

    lengths = [int(length) for length in attention_mask.sum(dim=-1).tolist()]
    cu_seqlens = F.pad(
        attention_mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0),
        (1, 0),
    )
    compact = torch.cat(
        [features[:length] for length, features in zip(lengths, text_features.unbind(0))],
        dim=0,
    )
    return compact, lengths, cu_seqlens, max(lengths)


def encode_reference_concepts(
    model,
    text_tokenizer,
    text_encoder,
    concepts: Sequence[str],
    include_eos_token: bool = True,
) -> list[torch.Tensor]:
    """Encode pooled reference concepts for the image-token mask.

    This mirrors the legacy ``token_aggregation_mode='mean_norm'`` path:
    valid text tokens are projected independently, then averaged in the
    ``text_proj_for_ca`` output space *before* each Cross-Attention layer
    applies its own key projection and normalization.
    """

    encoder_device = next(text_encoder.parameters()).device
    reference_concept_embeddings = []
    for concept in concepts:
        tokens = text_tokenizer(
            text=[concept],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(encoder_device, non_blocking=True)
        attention_mask = tokens.attention_mask.to(encoder_device, non_blocking=True)
        with torch.no_grad():
            features = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].float()[0]

        valid_length = int(attention_mask[0].sum().item())
        if not include_eos_token:
            valid_length = max(valid_length - 1, 1)
        features = features[:valid_length]
        proj_dtype = model.text_proj_for_ca[0].weight.dtype
        proj_device = model.text_proj_for_ca[0].weight.device
        token_embeddings = []
        # Keep the original implementation's per-token projection order. A
        # batched Linear/GELU call can select a different CUDA kernel and alter
        # the bf16 values used by the token mask.
        for token_index in range(features.shape[0]):
            token_features = features[token_index : token_index + 1].to(
                device=proj_device,
                dtype=proj_dtype,
            )
            with torch.no_grad():
                token_embeddings.append(
                    model.text_proj_for_ca(model.text_norm(token_features)).squeeze(0)
                )
        reference_concept_embeddings.append(
            torch.stack(token_embeddings, dim=0).mean(dim=0, keepdim=True).float()
        )
    return reference_concept_embeddings


def run_scaleerasure_inference(
    model,
    vae,
    text_tokenizer,
    text_encoder,
    prompt: str,
    *,
    scale_schedule,
    vae_type: int,
    seed: int | None,
    cfg_scale: float | list[float] = 3.0,
    temperature: float | list[float] = 0.5,
    top_k: int = 1,
    top_p: float = 0.97,
    safe_concept: str = "a person wearing clothed",
    unsafe_concept: str | None = None,
    reference_concepts: list[str] | None = None,
    token_mask_threshold: float = 0.332,
    bit_channel_threshold: float = 0.1,
    unsafe_penalty_weight: float = 3.0,
    safe_guidance_warmup_steps: int = 2,
    scale_start: int = 0,
    scale_end: int | None = 8,
    token_selection_layers: list[int] | None = None,
    token_selection_scales: list[int] | None = None,
    include_eos_token: bool = True,
    mask_snapshots: list[TokenMaskSnapshot] | None = None,
) -> torch.Tensor:
    """Run ScaleErasure selective logits guidance for one prompt.

    ``mask_snapshots`` is an optional list populated with the image-token
    relevance and binary masks used during generation.
    """

    if not isinstance(cfg_scale, list):
        cfg_scale = [cfg_scale] * len(scale_schedule)
    if not isinstance(temperature, list):
        temperature = [temperature] * len(scale_schedule)
    reference_concepts = reference_concepts or [unsafe_concept or "concept"]
    token_selection_layers = token_selection_layers or [1, 2, 3, 4, 5]
    token_selection_scales = token_selection_scales or list(range(len(scale_schedule)))
    unsafe_concept = unsafe_concept or reference_concepts[0]

    prompt_condition = encode_prompt(text_tokenizer, text_encoder, prompt)
    safe_condition = encode_prompt(text_tokenizer, text_encoder, safe_concept)
    with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        # The old implementation encoded reference concepts inside the same
        # bf16 autocast region as the autoregressive pass.
        reference_concept_embeddings = encode_reference_concepts(
            model,
            text_tokenizer,
            text_encoder,
            reference_concepts,
            include_eos_token=include_eos_token,
        )
        # The legacy runner encodes erase_concept after the reference tokens,
        # still inside that outer autocast region. Prompt and safety conditions
        # are encoded before it; preserving this asymmetry matches its 4-way
        # logits path.
        unsafe_condition = encode_prompt(text_tokenizer, text_encoder, unsafe_concept)
        return autoregressive_infer_scaleerasure(
            model=model,
            vae=vae,
            scale_schedule=scale_schedule,
            prompt_condition=prompt_condition,
            safe_condition=safe_condition,
            unsafe_condition=unsafe_condition,
            reference_concept_embeddings=reference_concept_embeddings,
            seed=seed,
            cfg_scale=cfg_scale,
            temperature=temperature,
            token_mask_threshold=token_mask_threshold,
            bit_channel_threshold=bit_channel_threshold,
            unsafe_penalty_weight=unsafe_penalty_weight,
            safe_guidance_warmup_steps=safe_guidance_warmup_steps,
            token_selection_layers=token_selection_layers,
            token_selection_scales=token_selection_scales,
            scale_start=scale_start,
            scale_end=scale_end,
            top_k=top_k,
            top_p=top_p,
            vae_type=vae_type,
            mask_snapshots=mask_snapshots,
        )


__all__ = [
    "encode_prompt",
    "encode_reference_concepts",
    "load_tokenizer",
    "load_transformer",
    "load_visual_tokenizer",
    "run_scaleerasure_inference",
]
