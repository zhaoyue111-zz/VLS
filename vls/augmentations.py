from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


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


def gaussian_blur_augment(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur over spatial axes without mixing channels."""
    x = image.astype(np.float32, copy=True)
    if x.ndim == 4:
        blur_sigma = (0.0, float(sigma), float(sigma), float(sigma))
    elif x.ndim == 3:
        blur_sigma = (float(sigma), float(sigma), float(sigma))
    else:
        raise ValueError(f"Expected 3D or channel-first 4D image, got shape {x.shape}")
    return gaussian_filter(x, sigma=blur_sigma, mode="nearest").astype(np.float32, copy=False)
