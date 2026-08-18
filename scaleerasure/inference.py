"""Model loading and the single public ScaleErasure generation call."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer

from .backend import run_scaleerasure_inference
from .config import ModelConfig, ScaleErasureConfig
from .io import save_image
from .sampler import TokenMaskSnapshot


@dataclass
class InferenceBundle:
    """Loaded components shared by all prompts in one run."""

    args: argparse.Namespace
    tokenizer: object
    text_encoder: torch.nn.Module
    vae: torch.nn.Module
    infinity: torch.nn.Module
    scale_schedule: list[tuple[int, int, int]]


def _resolve_model_paths(config: ModelConfig) -> tuple[Path, Path, Path]:
    model_root = config.model_root.expanduser()
    if config.model_type == "infinity_2b":
        default_model = model_root / "infinity_2b_reg.pth"
        default_vae = model_root / "infinity_vae_d32reg.pth"
    else:
        default_model = model_root / "infinity_8b_weights"
        default_vae = model_root / "infinity_vae_d56_f8_14_patchify.pth"

    text_encoder = config.text_encoder or Path(
        os.environ.get("SCALEERASURE_TEXT_ENCODER", str(model_root / "flan-t5-xl"))
    )
    model_path = config.model_path or default_model
    vae_path = config.vae_path or default_vae
    return text_encoder, model_path, vae_path


def _build_loader_args(config: ModelConfig) -> argparse.Namespace:
    text_encoder, model_path, vae_path = _resolve_model_paths(config)
    is_8b = config.model_type == "infinity_8b"
    return argparse.Namespace(
        pn="1M",
        model_path=str(model_path),
        cfg_insertion_layer=0,
        vae_type=14 if is_8b else 32,
        vae_path=str(vae_path),
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type=config.model_type,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=str(text_encoder),
        text_channels=2048,
        apply_spatial_patchify=int(is_8b),
        h_div_w_template=1.0,
        use_flex_attn=0,
        cache_dir=str(config.cache_dir),
        checkpoint_type="torch_shard" if is_8b else "torch",
        seed=0,
        bf16=1,
        enable_model_cache=False,
    )


def load_infinity_model(config: ModelConfig) -> InferenceBundle:
    """Load T5, the visual tokenizer, and Infinity once per run."""

    args = _build_loader_args(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt, device=device)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    for module in (text_encoder, vae, infinity):
        module.eval()
        module.requires_grad_(False)

    h_div_w = 1.0
    template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    raw_schedule = dynamic_resolution_h_w[template][args.pn]["scales"]
    scale_schedule = [(1, height, width) for _, height, width in raw_schedule]

    return InferenceBundle(
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        infinity=infinity,
        scale_schedule=scale_schedule,
    )


def generate_scaleerasure_image(
    bundle: InferenceBundle,
    prompt: str,
    method: ScaleErasureConfig,
    seed: int,
    mask_snapshots: list[TokenMaskSnapshot] | None = None,
) -> torch.Tensor:
    """Generate one image with the paper's ScaleErasure intervention.

    Pass a list through ``mask_snapshots`` only for diagnostics or
    visualisation; the default generation path does not retain token masks.
    """

    return run_scaleerasure_inference(
        bundle.infinity,
        bundle.vae,
        bundle.tokenizer,
        bundle.text_encoder,
        prompt,
        cfg_scale=method.cfg_scale,
        temperature=method.temperature,
        scale_schedule=bundle.scale_schedule,
        top_k=method.top_k,
        top_p=method.top_p,
        vae_type=bundle.args.vae_type,
        seed=seed,
        safe_concept=method.safe_concept,
        reference_concepts=list(method.reference_concepts),
        include_eos_token=True,
        token_mask_threshold=method.token_mask_threshold,
        token_selection_layers=list(method.token_selection_layers),
        token_selection_scales=list(method.token_selection_scales),
        unsafe_concept=method.resolved_unsafe_concept,
        bit_channel_threshold=method.bit_channel_threshold,
        unsafe_penalty_weight=method.unsafe_penalty_weight,
        safe_guidance_warmup_steps=method.safe_guidance_warmup_steps,
        scale_start=method.scale_start,
        scale_end=method.scale_end,
        mask_snapshots=mask_snapshots,
    )


__all__ = [
    "InferenceBundle",
    "generate_scaleerasure_image",
    "load_infinity_model",
    "save_image",
]
