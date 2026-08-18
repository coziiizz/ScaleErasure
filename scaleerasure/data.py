"""Dataset adapters used by the ScaleErasure runner."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_cached_arrow_dataset(path: str | Path) -> Dataset:
    """Load one Arrow shard or all shards in a local Arrow directory."""

    cache_path = Path(path).expanduser()
    if cache_path.is_file():
        shard_paths = [cache_path]
    elif cache_path.is_dir():
        shard_paths = sorted(cache_path.glob("*.arrow"))
    else:
        raise FileNotFoundError(f"Local Arrow dataset path does not exist: {cache_path}")

    if not shard_paths:
        raise FileNotFoundError(f"No .arrow shards found under: {cache_path}")
    shards = [Dataset.from_file(str(shard)) for shard in shard_paths]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def _load_dataset_or_arrow(
    arrow_env: str,
    dataset_name: str,
    split: str,
    cache_dir: Path,
) -> Dataset:
    arrow_path = os.environ.get(arrow_env)
    if arrow_path:
        return load_cached_arrow_dataset(arrow_path)
    return load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))


def load_i2p_prompts(cache_dir: Path, max_samples: int | None = None) -> list[PromptRecord]:
    """Return sexual-category prompts from the I2P training split."""

    dataset = _load_dataset_or_arrow(
        "SCALEERASURE_I2P_ARROW",
        "AIML-TUDA/i2p",
        "train",
        cache_dir,
    )
    records: list[PromptRecord] = []
    for row in dataset:
        categories = row.get("categories", [])
        if isinstance(categories, str):
            categories = [item.strip() for item in categories.split(",")]
        if "sexual" not in {str(item).strip() for item in categories}:
            continue

        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        records.append(PromptRecord(prompt, {"category": "sexual"}))
        if max_samples is not None and len(records) >= max_samples:
            break
    return records


def load_coco_prompts(
    cache_dir: Path,
    num_samples: int,
    max_samples: int | None = None,
) -> list[PromptRecord]:
    """Return captions from the paper's COCO validation subset."""

    count = num_samples if max_samples is None else min(num_samples, max_samples)
    arrow_path = os.environ.get("SCALEERASURE_COCO_ARROW")
    if arrow_path:
        dataset = load_cached_arrow_dataset(arrow_path)
        dataset = dataset.select(range(min(count, len(dataset))))
    else:
        dataset = load_dataset(
            "sayakpaul/coco-30-val-2014",
            split=f"train[:{count}]",
            cache_dir=str(cache_dir),
        )

    records = []
    for row in dataset:
        caption = str(row.get("caption", "")).strip()
        if caption:
            records.append(PromptRecord(caption, {"dataset": "coco"}))
    return records


def load_prompt_file(path: Path) -> list[PromptRecord]:
    """Load prompts from TXT, JSON, or CSV files."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("prompts", data.get("prompt", []))
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("JSON prompt files must contain a list or a 'prompts' field")
        return [PromptRecord(str(item).strip()) for item in data if str(item).strip()]

    if suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(path)
        column = "prompt" if "prompt" in frame.columns else "caption"
        if column not in frame.columns:
            raise ValueError("CSV prompt files must contain a 'prompt' or 'caption' column")
        records = []
        for _, row in frame.iterrows():
            value = str(row[column]).strip()
            if not value:
                continue
            metadata = {}
            if "concept" in frame.columns and str(row.get("concept", "")).strip():
                metadata["concept"] = str(row["concept"]).strip()
            records.append(PromptRecord(value, metadata))
        return records

    return [
        PromptRecord(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_prompts(config: Any) -> list[PromptRecord]:
    """Dispatch to the adapter selected by ``DatasetConfig``."""

    if config.experiment == "i2p":
        return load_i2p_prompts(config.dataset_cache, config.max_samples)
    if config.experiment == "coco":
        return load_coco_prompts(
            config.dataset_cache,
            config.coco_num_samples,
            config.max_samples,
        )
    if config.experiment == "case":
        if config.prompts_file is None:
            raise ValueError("case experiment requires --prompts_file")
        records = load_prompt_file(config.prompts_file)
        return records if config.max_samples is None else records[: config.max_samples]
    if config.prompt is None or not config.prompt.strip():
        raise ValueError("prompt experiment requires --prompt")
    return [PromptRecord(config.prompt.strip())]
