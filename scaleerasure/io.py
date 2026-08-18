"""Input/output helpers for generated images."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def save_image(path: str, image: torch.Tensor | np.ndarray) -> None:
    """Save one Infinity output tensor as an RGB PNG."""

    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"save_image expects one image, got shape={array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB/BGR image, got shape={array.shape}")

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    # Infinity returns BGR tensors for compatibility with its upstream tools.
    Image.fromarray(array[..., ::-1]).save(path)


__all__ = ["save_image"]
