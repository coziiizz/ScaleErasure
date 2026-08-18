"""Export image-token relevance and masks for visual inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .sampler import TokenMaskSnapshot


def _first_frame_grid(values: np.ndarray, scale_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert flattened ``t, h, w`` token values to the legacy ``t=0`` view."""

    expected = int(np.prod(scale_shape))
    if values.size != expected:
        raise ValueError(
            f"token values have {values.size} entries, expected {expected} for "
            f"scale_shape={scale_shape}"
        )
    return values.reshape(scale_shape)[0]


def _heatmap(
    values: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Map relevance values to a blue-green-yellow-red palette."""

    anchors = np.asarray(
        [[32, 48, 160], [30, 150, 180], [245, 230, 60], [210, 35, 45]],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(anchors))
    if not lower < upper:
        raise ValueError(f"heatmap range must be increasing, got [{lower}, {upper}]")
    clipped = np.clip(values.astype(np.float32), lower, upper)
    clipped = (clipped - lower) / (upper - lower)
    channels = [np.interp(clipped, positions, anchors[:, channel]) for channel in range(3)]
    return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)


def _resize_nearest(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Resize a mask or heatmap without smoothing token boundaries."""

    return Image.fromarray(array).resize(size, resample=Image.Resampling.NEAREST)


def _save_overlay(
    image_path: Path,
    binary_grid: np.ndarray,
    output_path: Path,
    alpha: float = 0.48,
) -> None:
    """Overlay selected image tokens in red on the generated RGB image."""

    image = Image.open(image_path).convert("RGB")
    full_mask = _resize_nearest(np.where(binary_grid > 0, 255, 0).astype(np.uint8), image.size)
    image_array = np.asarray(image).astype(np.float32)
    mask = np.asarray(full_mask) > 0
    image_array[mask] = (1.0 - alpha) * image_array[mask] + alpha * np.asarray([255.0, 32.0, 32.0])
    Image.fromarray(np.rint(image_array).astype(np.uint8)).save(output_path)


def save_token_mask_artifacts(
    output_dir: str | Path,
    snapshot: TokenMaskSnapshot,
    *,
    image_path: str | Path,
    prompt: str,
    panel_size: int = 512,
) -> dict[str, Path]:
    """Save raw arrays, metadata, heatmap, binary mask, and image overlay.

    The PNGs use the first temporal plane (``t=0``), matching the historical
    implementation's mask visualisation.  The compressed NumPy archive keeps
    the complete flattened mask and relevance vector for exact inspection.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    scale_tag = f"scale{snapshot.scale_index:02d}"

    relevance = snapshot.relevance.detach().cpu().numpy().astype(np.float32)
    binary_mask = snapshot.binary_mask.detach().cpu().numpy().astype(np.uint8)
    relevance_grid = _first_frame_grid(relevance, snapshot.scale_shape)
    binary_grid = _first_frame_grid(binary_mask, snapshot.scale_shape)

    heatmap_lower = max(0.0, snapshot.threshold - 0.01)
    heatmap_upper = min(1.0, snapshot.threshold + 0.01)
    heatmap = _resize_nearest(
        _heatmap(relevance_grid, heatmap_lower, heatmap_upper),
        (panel_size, panel_size),
    )
    binary_image = _resize_nearest(
        np.where(binary_grid > 0, 255, 0).astype(np.uint8),
        (panel_size, panel_size),
    ).convert("L")
    heatmap_path = output_path / f"token_relevance_{scale_tag}.png"
    binary_path = output_path / f"token_mask_{scale_tag}.png"
    overlay_path = output_path / f"token_mask_overlay_{scale_tag}.png"
    panel_path = output_path / f"token_mask_panel_{scale_tag}.png"
    heatmap.save(heatmap_path)
    binary_image.save(binary_path)
    _save_overlay(image_path, binary_grid, overlay_path)

    panel = Image.new("RGB", (panel_size * 3, panel_size))
    panel.paste(Image.open(image_path).convert("RGB").resize((panel_size, panel_size)), (0, 0))
    panel.paste(heatmap.convert("RGB"), (panel_size, 0))
    panel.paste(binary_image.convert("RGB"), (panel_size * 2, 0))
    panel.save(panel_path)

    archive_path = output_path / f"token_mask_{scale_tag}.npz"
    np.savez_compressed(
        archive_path,
        relevance=relevance,
        binary_mask=binary_mask,
        relevance_t0=relevance_grid,
        binary_mask_t0=binary_grid,
    )

    metadata = {
        "prompt": prompt,
        "scale_index": snapshot.scale_index,
        "scale_shape": list(snapshot.scale_shape),
        "threshold": snapshot.threshold,
        "heatmap_range": [heatmap_lower, heatmap_upper],
        "selected_layers": list(snapshot.selected_layers),
        "num_tokens": int(binary_mask.size),
        "active_tokens": int(binary_mask.sum()),
        "coverage": float(binary_mask.mean()),
        "first_frame_active_tokens": int(binary_grid.sum()),
        "first_frame_coverage": float(binary_grid.mean()),
        "visualization_note": (
            "PNG files show the first temporal plane (t=0); NPZ keeps all t,h,w tokens."
        ),
    }
    metadata_path = output_path / f"token_mask_{scale_tag}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "heatmap": heatmap_path,
        "binary_mask": binary_path,
        "overlay": overlay_path,
        "panel": panel_path,
        "archive": archive_path,
        "metadata": metadata_path,
    }


__all__ = ["save_token_mask_artifacts"]
