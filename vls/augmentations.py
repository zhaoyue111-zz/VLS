from __future__ import annotations

import numpy as np


def gamma_augment(image: np.ndarray, gamma: float = 1.2, eps: float = 1e-6) -> np.ndarray:
    """Apply monotonic gamma augmentation after min-max normalization per volume."""
    x = image.astype(np.float32, copy=True)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return x
    normalized = (x - lo) / (hi - lo)
    transformed = np.power(np.clip(normalized, 0.0, 1.0), gamma)
    return transformed * (hi - lo) + lo


def contrast_augment(image: np.ndarray, strength: float, eps: float = 1e-6) -> np.ndarray:
    """Adjust contrast around the image mean and clip to the original intensity range."""
    x = image.astype(np.float32, copy=True)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return x
    mean = float(np.mean(x))
    contrast_factor = 1.0 + float(strength)
    adjusted = mean + contrast_factor * (x - mean)
    return np.clip(adjusted, lo, hi)
