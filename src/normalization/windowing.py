"""
CT Hounsfield Unit (HU) windowing and intensity normalization.

CT scans have a wide dynamic range (-1000 to +3000 HU). Windowing
selects a sub-range of clinical interest and maps it to 0-255 for
display or downstream processing. Different anatomical structures
require different window presets.
"""

import numpy as np
from typing import Tuple, Optional


# Standard CT window presets — (window_center, window_width) in HU
WINDOW_PRESETS = {
    "lung":         (-600, 1500),    # Lung parenchyma
    "mediastinum":  (50, 350),       # Soft tissue in chest
    "bone":         (400, 1800),     # Bone structures
    "brain":        (40, 80),        # Brain tissue
    "abdomen":      (40, 400),       # Abdominal soft tissue
    "liver":        (60, 160),       # Liver parenchyma
    "stroke":       (40, 40),        # Acute stroke detection
    "soft_tissue":  (50, 250),       # General soft tissue
}


def window_ct(
    image: np.ndarray,
    window_preset: Optional[str] = None,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    output_range: Tuple[int, int] = (0, 255),
) -> np.ndarray:
    """
    Apply CT windowing to a Hounsfield Unit (HU) image.

    Args:
        image: CT image array in HU (any shape)
        window_preset: Named preset from WINDOW_PRESETS
        window_center: Custom window center (HU)
        window_width: Custom window width (HU)
        output_range: Output intensity range (default 0-255)

    Returns:
        Windowed image as float32, scaled to output_range
    """
    if window_preset is not None:
        if window_preset not in WINDOW_PRESETS:
            raise ValueError(
                f"Unknown preset '{window_preset}'. "
                f"Available: {list(WINDOW_PRESETS.keys())}"
            )
        window_center, window_width = WINDOW_PRESETS[window_preset]

    if window_center is None or window_width is None:
        raise ValueError("Must provide either window_preset or (center, width)")

    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2

    windowed = np.clip(image, img_min, img_max)

    # Scale to output range
    out_min, out_max = output_range
    windowed = (windowed - img_min) / (img_max - img_min)
    windowed = windowed * (out_max - out_min) + out_min

    return windowed.astype(np.float32)


def z_score_normalize(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalize an image: (x - mean) / std.

    Args:
        image: Input image array
        mask: Optional boolean mask — only compute statistics within mask
        epsilon: Small constant to avoid division by zero

    Returns:
        Normalized image (zero mean, unit variance)
    """
    if mask is not None:
        values = image[mask > 0]
    else:
        values = image

    mean = float(np.mean(values))
    std = float(np.std(values))

    return (image - mean) / (std + epsilon)


def min_max_normalize(
    image: np.ndarray,
    output_range: Tuple[float, float] = (0.0, 1.0),
    clip_percentiles: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Min-max normalize an image to a target range.

    Args:
        image: Input image array
        output_range: Target range (default 0-1)
        clip_percentiles: Optional (low, high) percentiles for robust normalization

    Returns:
        Normalized image
    """
    if clip_percentiles is not None:
        low, high = np.percentile(image, clip_percentiles)
        image = np.clip(image, low, high)
        img_min, img_max = low, high
    else:
        img_min = float(image.min())
        img_max = float(image.max())

    if img_max - img_min < 1e-8:
        return np.zeros_like(image)

    out_min, out_max = output_range
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (out_max - out_min) + out_min

    return normalized.astype(np.float32)


def percentile_normalize(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """
    Robust normalization clipping at low/high percentiles.

    Useful for MRI where intensity ranges vary across scanners.

    Args:
        image: Input image array
        lower: Lower percentile (default: 1.0)
        upper: Upper percentile (default: 99.0)

    Returns:
        Normalized image in [0, 1]
    """
    return min_max_normalize(image, clip_percentiles=(lower, upper))


if __name__ == "__main__":
    print("CT Windowing presets:")
    for name, (center, width) in WINDOW_PRESETS.items():
        print(f"  {name:14s}: center={center:>5}, width={width:>5}")

    # Sanity check
    dummy_ct = np.random.randint(-1000, 3000, (512, 512)).astype(np.float32)
    lung_windowed = window_ct(dummy_ct, window_preset="lung")
    print(f"\nLung windowed range: [{lung_windowed.min():.1f}, {lung_windowed.max():.1f}]")

    z_norm = z_score_normalize(dummy_ct)
    print(f"Z-score mean: {z_norm.mean():.4f}, std: {z_norm.std():.4f}")
