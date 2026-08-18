#!/usr/bin/env python3
"""Generate one I2P and one COCO-like sample with token-mask artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaleerasure.config import ModelConfig, ScaleErasureConfig  # noqa: E402
from scaleerasure.data import load_i2p_prompts  # noqa: E402
from scaleerasure.inference import generate_scaleerasure_image, load_infinity_model  # noqa: E402
from scaleerasure.io import save_image  # noqa: E402
from scaleerasure.mask_export import save_token_mask_artifacts  # noqa: E402
from scaleerasure.sampler import TokenMaskSnapshot  # noqa: E402


def _path_from_env(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def _load_method_config(path: Path) -> tuple[ModelConfig, ScaleErasureConfig]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    model_data = config.get("model", {})
    method_data = dict(config.get("method", {}))
    # ``seed`` belongs to the run, while ScaleErasureConfig contains only
    # method parameters.  The CLI keeps the run seed explicit below.
    method_data.pop("seed", None)
    model = ModelConfig(
        model_type=str(model_data.get("type", "infinity_2b")),
        model_root=_path_from_env("SCALEERASURE_MODEL_ROOT", "weights"),
        text_encoder=_path_from_env(
            "SCALEERASURE_TEXT_ENCODER",
            str(_path_from_env("SCALEERASURE_MODEL_ROOT", "weights") / "flan-t5-xl"),
        ),
    )
    return model, ScaleErasureConfig(**method_data)


def _select_snapshot(snapshots: list[TokenMaskSnapshot], scale_index: int) -> TokenMaskSnapshot:
    for snapshot in snapshots:
        if snapshot.scale_index == scale_index:
            return snapshot
    available = [snapshot.scale_index for snapshot in snapshots]
    raise RuntimeError(
        f"No token mask captured at scale={scale_index}; available scales={available}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ScaleErasure token masks for one I2P and one COCO prompt."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scaleerasure_i2p.yaml")
    parser.add_argument(
        "--dataset_cache",
        type=Path,
        default=_path_from_env("SCALEERASURE_DATASET_CACHE", ".cache/datasets"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "outputs/token_mask_examples",
    )
    parser.add_argument("--i2p_index", type=int, default=0)
    parser.add_argument(
        "--i2p_prompt",
        default=None,
        help="Override the sexual I2P prompt instead of reading the local dataset.",
    )
    parser.add_argument(
        "--coco_prompt",
        default="A bicycle replica with a clock as the front wheel.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale",
        type=int,
        default=7,
        help="Active intervention scale to export; 7 is the last scale before early exit 8.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    model_config, method_config = _load_method_config(config_path.resolve())
    dataset_cache = args.dataset_cache.expanduser()
    if not dataset_cache.is_absolute():
        dataset_cache = ROOT / dataset_cache

    if args.i2p_prompt is None:
        prompts = load_i2p_prompts(dataset_cache, max_samples=args.i2p_index + 1)
        if args.i2p_index >= len(prompts):
            raise IndexError(
                f"I2P index {args.i2p_index} is unavailable; loaded {len(prompts)} prompts"
            )
        i2p_prompt = prompts[args.i2p_index].prompt
    else:
        i2p_prompt = args.i2p_prompt.strip()
    if not i2p_prompt:
        raise ValueError("I2P prompt must not be empty")
    if not args.coco_prompt.strip():
        raise ValueError("COCO prompt must not be empty")

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_infinity_model(model_config)

    cases = (
        ("i2p_sexual", i2p_prompt),
        ("coco_normal", args.coco_prompt.strip()),
    )
    for name, prompt in cases:
        print(f"[MaskExport] case={name}")
        print(f"[MaskExport] prompt={prompt}")
        snapshots: list[TokenMaskSnapshot] = []
        image = generate_scaleerasure_image(
            bundle,
            prompt,
            method_config,
            seed=args.seed,
            mask_snapshots=snapshots,
        )
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = case_dir / "image.png"
        save_image(str(image_path), image)
        snapshot = _select_snapshot(snapshots, args.scale)
        artifacts = save_token_mask_artifacts(
            case_dir,
            snapshot,
            image_path=image_path,
            prompt=prompt,
        )
        print(
            f"[MaskExport] scale={snapshot.scale_index}, shape={snapshot.scale_shape}, "
            f"active={int(snapshot.binary_mask.sum())}/{snapshot.binary_mask.numel()}"
        )
        for artifact_name, artifact_path in artifacts.items():
            print(f"[MaskExport] {artifact_name}={artifact_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
