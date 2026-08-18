"""Command-line entry point for ScaleErasure generation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm

from .config import (
    DEFAULT_REFERENCE_CONCEPTS,
    DatasetConfig,
    ModelConfig,
    ScaleErasureConfig,
)
from .data import load_prompts
from .inference import generate_scaleerasure_image, load_infinity_model, save_image


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def _parse_concepts(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split("|") if item.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError(
            "reference concepts must be a JSON list or a '|' separated string"
        )
    return tuple(str(item).strip() for item in parsed if str(item).strip())


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-time concept erasure with ScaleErasure."
    )
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment",
        choices=("i2p", "coco", "case", "prompt"),
        default="i2p",
    )
    parser.add_argument(
        "--model_type",
        choices=("infinity_2b", "infinity_8b"),
        default="infinity_2b",
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=_env_path("SCALEERASURE_MODEL_ROOT", "weights"),
    )
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--vae_path", type=Path, default=None)
    parser.add_argument(
        "--text_encoder",
        type=Path,
        default=None,
    )
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache"))

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_env_path("SCALEERASURE_OUTPUT_DIR", "outputs"),
    )
    parser.add_argument(
        "--dataset_cache",
        type=Path,
        default=_env_path("SCALEERASURE_DATASET_CACHE", ".cache/datasets"),
    )
    parser.add_argument("--prompts_file", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--coco_num_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--safe_concept", default="a person wearing clothed")
    parser.add_argument("--unsafe_concept", default=None)
    parser.add_argument(
        "--reference_concepts",
        type=_parse_concepts,
        default=DEFAULT_REFERENCE_CONCEPTS,
        help="JSON list or '|' separated concepts; the first is the unsafe target.",
    )
    parser.add_argument("--token_mask_threshold", type=float, default=0.332)
    parser.add_argument("--bit_channel_threshold", type=float, default=0.1)
    parser.add_argument("--unsafe_penalty_weight", type=float, default=3.0)
    parser.add_argument("--safe_guidance_warmup_steps", type=int, default=2)
    parser.add_argument("--scale_start", type=int, default=0)
    parser.add_argument("--scale_end", type=int, default=8)
    parser.add_argument(
        "--token_selection_layers",
        type=_parse_int_list,
        default=(1, 2, 3, 4, 5),
    )
    parser.add_argument(
        "--token_selection_scales",
        type=_parse_int_list,
        default=tuple(range(13)),
    )
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.97)
    return parser


def _build_configs(
    args: argparse.Namespace,
) -> tuple[DatasetConfig, ModelConfig, ScaleErasureConfig]:
    dataset = DatasetConfig(
        experiment=args.experiment,
        dataset_cache=args.dataset_cache,
        prompts_file=args.prompts_file,
        prompt=args.prompt,
        max_samples=args.max_samples,
        coco_num_samples=args.coco_num_samples,
    )
    model = ModelConfig(
        model_type=args.model_type,
        model_root=args.model_root,
        text_encoder=args.text_encoder,
        model_path=args.model_path,
        vae_path=args.vae_path,
        cache_dir=args.cache_dir,
    )
    method = ScaleErasureConfig(
        safe_concept=args.safe_concept,
        unsafe_concept=args.unsafe_concept,
        reference_concepts=args.reference_concepts,
        token_mask_threshold=args.token_mask_threshold,
        bit_channel_threshold=args.bit_channel_threshold,
        unsafe_penalty_weight=args.unsafe_penalty_weight,
        safe_guidance_warmup_steps=args.safe_guidance_warmup_steps,
        scale_start=args.scale_start,
        scale_end=args.scale_end,
        token_selection_layers=args.token_selection_layers,
        token_selection_scales=args.token_selection_scales,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    return dataset, model, method


def _write_run_metadata(
    output_dir: Path,
    dataset: DatasetConfig,
    model: ModelConfig,
    method: ScaleErasureConfig,
) -> None:
    payload = {
        "dataset": asdict(dataset),
        "model": asdict(model),
        "method": asdict(method),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset, model, method = _build_configs(args)
    output_dir = args.output_dir.expanduser()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(dataset)
    if not prompts:
        raise RuntimeError(f"No prompts found for experiment={dataset.experiment}")

    print(f"[ScaleErasure] experiment={dataset.experiment}, samples={len(prompts)}")
    print(f"[ScaleErasure] token_mask_threshold={method.token_mask_threshold}")
    print(f"[ScaleErasure] reference_concepts={list(method.reference_concepts)}")
    _write_run_metadata(output_dir, dataset, model, method)

    bundle = load_infinity_model(model)
    records = []
    for index, record in enumerate(tqdm(prompts, desc="Generating")):
        print(f"[{index + 1}/{len(prompts)}] {record.prompt}")
        image = generate_scaleerasure_image(bundle, record.prompt, method, seed=args.seed)
        image_path = image_dir / f"{index:05d}.png"
        save_image(str(image_path), image)
        records.append(
            {
                "index": index,
                "prompt": record.prompt,
                "image": str(image_path.relative_to(output_dir)),
                **record.metadata,
            }
        )

    pd.DataFrame(records).to_csv(output_dir / "prompts.csv", index=False)
    print(f"[ScaleErasure] saved {len(records)} images to {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
