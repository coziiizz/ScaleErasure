"""Configuration objects for the public ScaleErasure inference path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REFERENCE_CONCEPTS = (
    "nudity, sexual, explicit, porn, erotic, fetish",
    "clothed",
    "background, object, animal, tree, sky, architecture",
)


@dataclass(frozen=True)
class ScaleErasureConfig:
    """Parameters that define one ScaleErasure generation run.

    ``reference_concepts[0]`` is the unsafe target concept. The remaining
    prompts are reference concepts used for token-level relevance comparison;
    they do not have independent thresholds.
    """

    # Keep the historical I2P benchmark prompt verbatim.  The old runner
    # uses "clothed" (rather than the grammatically nicer "clothes"), and
    # changing it changes the safety branch's text conditioning.
    safe_concept: str = "a person wearing clothed"
    unsafe_concept: str | None = None
    reference_concepts: tuple[str, ...] = field(default_factory=lambda: DEFAULT_REFERENCE_CONCEPTS)
    token_mask_threshold: float = 0.332
    bit_channel_threshold: float = 0.1
    unsafe_penalty_weight: float = 3.0
    safe_guidance_warmup_steps: int = 2
    # The legacy implementation evaluates the intervention path from scale
    # zero.  Its first two scales are effectively unchanged by warmup.
    scale_start: int = 0
    scale_end: int | None = 8
    token_selection_layers: tuple[int, ...] = (1, 2, 3, 4, 5)
    token_selection_scales: tuple[int, ...] = tuple(range(13))
    cfg_scale: float = 3.0
    temperature: float = 0.5
    top_k: int = 1
    top_p: float = 0.97

    def __post_init__(self) -> None:
        concepts = tuple(str(item).strip() for item in self.reference_concepts if str(item).strip())
        if not concepts:
            raise ValueError("reference_concepts must contain at least one concept")
        object.__setattr__(self, "reference_concepts", concepts)

        if not 0.0 <= self.token_mask_threshold <= 1.0:
            raise ValueError("token_mask_threshold must be in [0, 1]")
        if not 0.0 <= self.bit_channel_threshold <= 1.0:
            raise ValueError("bit_channel_threshold must be in [0, 1]")
        if self.unsafe_penalty_weight < 0:
            raise ValueError("unsafe_penalty_weight must be non-negative")
        if self.safe_guidance_warmup_steps < 0:
            raise ValueError("safe_guidance_warmup_steps must be non-negative")
        if self.scale_start < 0:
            raise ValueError("scale_start must be non-negative")
        if self.scale_end is not None and self.scale_end < 0:
            raise ValueError("scale_end must be non-negative or None")
        if not self.token_selection_layers:
            raise ValueError("token_selection_layers must contain at least one layer index")
        if not self.token_selection_scales:
            raise ValueError("token_selection_scales must contain at least one scale index")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @property
    def resolved_unsafe_concept(self) -> str:
        return self.unsafe_concept or self.reference_concepts[0]


@dataclass(frozen=True)
class ModelConfig:
    """Paths and architecture settings needed to load Infinity."""

    model_type: str = "infinity_2b"
    model_root: Path = Path("weights")
    text_encoder: Path | None = None
    model_path: Path | None = None
    vae_path: Path | None = None
    cache_dir: Path = Path(".cache")

    def __post_init__(self) -> None:
        if self.model_type not in {"infinity_2b", "infinity_8b"}:
            raise ValueError(
                f"Unsupported model_type={self.model_type!r}; "
                "the public runner supports infinity_2b and infinity_8b"
            )
        object.__setattr__(self, "model_root", Path(self.model_root).expanduser())
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.text_encoder is not None:
            object.__setattr__(self, "text_encoder", Path(self.text_encoder).expanduser())
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        if self.vae_path is not None:
            object.__setattr__(self, "vae_path", Path(self.vae_path).expanduser())


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset selection for one generation run."""

    experiment: str
    dataset_cache: Path = Path(".cache/datasets")
    prompts_file: Path | None = None
    prompt: str | None = None
    max_samples: int | None = None
    coco_num_samples: int = 10_000

    def __post_init__(self) -> None:
        if self.experiment not in {"i2p", "coco", "case", "prompt"}:
            raise ValueError(f"Unknown experiment: {self.experiment}")
        object.__setattr__(self, "dataset_cache", Path(self.dataset_cache).expanduser())
        if self.prompts_file is not None:
            object.__setattr__(self, "prompts_file", Path(self.prompts_file).expanduser())
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive or None")
        if self.coco_num_samples < 1:
            raise ValueError("coco_num_samples must be positive")
