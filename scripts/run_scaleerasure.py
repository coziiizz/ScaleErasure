#!/usr/bin/env python3
"""Run a ScaleErasure experiment from a checked-in YAML configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _comma(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def _append_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(config: dict[str, Any]) -> list[str]:
    method = config.get("method", {})
    runtime = config.get("runtime", {})
    data = config.get("data", {})
    model = config.get("model", {})
    experiment = config["experiment"]

    command = [
        sys.executable,
        "-m",
        "scaleerasure.cli",
        "--exp",
        str(experiment),
        "--model_type",
        str(model.get("type", "infinity_2b")),
        "--seed",
        str(method.get("seed", 42)),
        "--safe_concept",
        str(method["safe_concept"]),
        "--reference_concepts",
        json.dumps(method["reference_concepts"], ensure_ascii=False),
        "--token_mask_threshold",
        str(method.get("token_mask_threshold", 0.332)),
        "--bit_channel_threshold",
        str(method.get("bit_channel_threshold", 0.1)),
        "--unsafe_penalty_weight",
        str(method.get("unsafe_penalty_weight", 3.0)),
        "--safe_guidance_warmup_steps",
        str(method.get("safe_guidance_warmup_steps", 2)),
        "--top_k",
        str(method.get("top_k", 1)),
        "--temperature",
        str(method.get("temperature", 0.5)),
        "--top_p",
        str(method.get("top_p", 0.97)),
        "--token_selection_layers",
        _comma(method.get("token_selection_layers", [1, 2, 3, 4, 5])),
        "--token_selection_scales",
        _comma(method.get("token_selection_scales", list(range(13)))),
        "--cfg_scale",
        str(method.get("cfg_scale", 3.0)),
    ]

    _append_optional(command, "--model_root", model.get("root", model.get("model_root")))
    _append_optional(command, "--model_path", model.get("path", model.get("model_path")))
    _append_optional(command, "--vae_path", model.get("vae_path"))
    _append_optional(command, "--text_encoder", model.get("text_encoder"))
    _append_optional(command, "--cache_dir", model.get("cache_dir"))

    if method.get("unsafe_concept"):
        command.extend(["--unsafe_concept", str(method["unsafe_concept"])])

    max_samples = runtime.get("max_samples")
    if max_samples is not None:
        command.extend(["--max_samples", str(max_samples)])

    if method.get("scale_end") is not None:
        command.extend(["--scale_end", str(method["scale_end"])])
    if method.get("scale_start") is not None:
        command.extend(["--scale_start", str(method["scale_start"])])
    if experiment == "coco":
        command.extend(["--coco_num_samples", str(data.get("coco_num_samples", 3))])

    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override the dataset limit for a short verification run.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = _expand_env(yaml.safe_load(handle))
    if not isinstance(config, dict) or "experiment" not in config:
        raise ValueError(f"Invalid ScaleErasure config: {config_path}")

    runtime = config.get("runtime", {})
    environment = os.environ.copy()
    output_dir = Path(runtime.get("output_dir", "outputs"))
    dataset_cache = Path(runtime.get("dataset_cache", ".cache/datasets"))
    environment.setdefault(
        "SCALEERASURE_OUTPUT_DIR",
        str((ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir),
    )
    environment.setdefault(
        "SCALEERASURE_DATASET_CACHE",
        str((ROOT / dataset_cache).resolve() if not dataset_cache.is_absolute() else dataset_cache),
    )
    command = build_command(config)
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    print("[ScaleErasure]", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
