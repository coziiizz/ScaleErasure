"""Minimal Infinity loaders used by the public ScaleErasure inference path."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast

from infinity.models.infinity import Infinity

torch._dynamo.config.cache_size_limit = 64


def load_tokenizer(
    t5_path: str | Path = "",
    device: str | torch.device | None = None,
) -> tuple[T5TokenizerFast, T5EncoderModel]:
    """Load the T5 tokenizer and frozen text encoder."""

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[Loading tokenizer and text encoder]")

    tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
        str(t5_path), revision=None, legacy=True
    )
    tokenizer.model_max_length = 512
    dtype = torch.float16 if target_device.type == "cuda" else torch.float32
    text_encoder = T5EncoderModel.from_pretrained(str(t5_path), torch_dtype=dtype)
    text_encoder.to(target_device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder


def save_slim_model(
    model_path: str | Path,
    save_file: str | Path | None = None,
    device: str | torch.device = "cpu",
    key: str = "gpt_fsdp",
) -> Path:
    """Extract an Infinity transformer state dict from a full checkpoint."""

    source_path = Path(model_path)
    target_path = (
        Path(save_file)
        if save_file
        else source_path.with_name(f"{source_path.stem}-slim{source_path.suffix}")
    )
    print(f"[Save slim model] {source_path} -> {target_path}")
    checkpoint = torch.load(source_path, map_location=device)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["trainer"][key], target_path)
    return target_path


def _load_infinity(
    *,
    rope2d_each_sa_layer: int,
    rope2d_normalized_by_hw: int,
    pn: str,
    use_bit_label: int,
    add_lvl_embeding_only_first_block: int,
    model_path: str | Path,
    vae: torch.nn.Module,
    device: torch.device,
    model_kwargs: dict[str, Any],
    text_channels: int,
    apply_spatial_patchify: int,
    use_flex_attn: int,
    bf16: int,
    checkpoint_type: str,
) -> Infinity:
    """Construct Infinity and load either a regular or sharded checkpoint."""

    print("[Loading Infinity]")
    autocast_enabled = device.type == "cuda"
    with (
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ),
        torch.no_grad(),
    ):
        model = Infinity(
            vae_local=vae,
            text_channels=text_channels,
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=None,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)

        parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e9
        print(f"[Infinity] model={parameter_count:.2f}B, bf16={bool(bf16)}")
        if bf16:
            for block in model.unregistered_blocks:
                block.bfloat16()

        model.eval()
        model.requires_grad_(False)

        checkpoint_path = str(model_path)
        if checkpoint_type == "torch":
            state_dict = torch.load(checkpoint_path, map_location=device)
            print(model.load_state_dict(state_dict))
        elif checkpoint_type == "torch_shard":
            from transformers.modeling_utils import load_sharded_checkpoint

            load_sharded_checkpoint(model, checkpoint_path, strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint_type={checkpoint_type!r}")

        model.rng = torch.Generator(device=device)
        return model


def load_visual_tokenizer(args: Namespace) -> torch.nn.Module:
    """Load the bitwise VAE selected by the Infinity loader arguments."""

    from infinity.models.bsq_vae.vae import vae_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.vae_type not in {14, 16, 18, 20, 24, 32, 64}:
        raise ValueError(f"vae_type={args.vae_type} is not supported")

    codebook_dim = args.vae_type
    if args.apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]

    vae = vae_model(
        args.vae_path,
        "dynamic",
        codebook_dim,
        2**codebook_dim,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        test_mode=True,
    )
    return vae.to(device).eval()


def load_transformer(vae: torch.nn.Module, args: Namespace) -> Infinity:
    """Load an Infinity-2B or Infinity-8B transformer for inference."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(args.model_path)
    if args.checkpoint_type == "torch":
        checkpoint_path = model_path
        if args.enable_model_cache:
            cache_dir = Path(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(cache_dir / Path(model_path).name)
            if not Path(checkpoint_path).exists():
                save_slim_model(model_path, checkpoint_path, device=device)
        print(f"[Infinity] checkpoint={checkpoint_path}")
    elif args.checkpoint_type == "torch_shard":
        checkpoint_path = model_path
    else:
        raise ValueError(f"Unsupported checkpoint_type={args.checkpoint_type!r}")

    model_shapes = {
        "infinity_2b": dict(
            depth=32,
            embed_dim=2048,
            num_heads=16,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=8,
        ),
        "infinity_8b": dict(
            depth=40,
            embed_dim=3584,
            num_heads=28,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=8,
        ),
    }
    if args.model_type not in model_shapes:
        raise ValueError(f"Unsupported model_type={args.model_type!r}")

    return _load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=checkpoint_path,
        vae=vae,
        device=device,
        model_kwargs=model_shapes[args.model_type],
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )


__all__ = ["load_tokenizer", "load_transformer", "load_visual_tokenizer"]
