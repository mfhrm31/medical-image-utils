"""
Region of Interest (ROI) extraction for medical images.

Provides utilities for isolating anatomical regions of interest
(lungs, tumors, organs) using thresholding and morphological operations.
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, filters
from typing import Tuple, Optional


def adaptive_threshold(
    image: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """
    Apply adaptive thresholding to extract foreground.

    Args:
        image: 2D or 3D image array
        method: Thresholding method ('otsu', 'li', 'yen')

    Returns:
        Binary mask of same shape as input
    """
    if method == "otsu":
        threshold = filters.threshold_otsu(image)
    elif method == "li":
        threshold = filters.threshold_li(image)
    elif method == "yen":
        threshold = filters.threshold_yen(image)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'otsu', 'li', or 'yen'.")

    return (image > threshold).astype(np.uint8)


def morphological_cleanup(
    mask: np.ndarray,
    closing_radius: int = 3,
    opening_radius: int = 2,
    min_size: int = 100,
) -> np.ndarray:
    """
    Clean up a binary mask using morphological operations.

    Applies (in order):
        1. Closing — fills small holes
        2. Opening — removes small noise
        3. Small-object removal — drops isolated regions below min_size

    Args:
        mask: Binary mask (2D or 3D)
        closing_radius: Radius for closing operation
        opening_radius: Radius for opening operation
        min_size: Minimum connected component size to keep

    Returns:
        Cleaned binary mask
    """
    if mask.ndim == 2:
        closing_selem = morphology.disk(closing_radius)
        opening_selem = morphology.disk(opening_radius)
    elif mask.ndim == 3:
        closing_selem = morphology.ball(closing_radius)
        opening_selem = morphology.ball(opening_radius)
    else:
        raise ValueError(f"Mask must be 2D or 3D, got {mask.ndim}D")

    cleaned = morphology.binary_closing(mask, closing_selem)
    cleaned = morphology.binary_opening(cleaned, opening_selem)
    cleaned = morphology.remove_small_objects(cleaned.astype(bool), min_size=min_size)

    return cleaned.astype(np.uint8)


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    Useful for isolating a single organ when multiple regions
    have similar intensity.

    Args:
        mask: Binary mask

    Returns:
        Binary mask containing only the largest component
    """
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    largest_label = int(np.argmax(component_sizes))

    return (labeled == largest_label).astype(np.uint8)


def extract_bbox(mask: np.ndarray, padding: int = 0) -> Tuple[slice, ...]:
    """
    Compute bounding box around a binary mask's foreground.

    Args:
        mask: Binary mask (2D or 3D)
        padding: Number of pixels to pad each side

    Returns:
        Tuple of slices for cropping the original image
    """
    coords = np.array(np.where(mask > 0))

    if coords.size == 0:
        raise ValueError("Mask contains no foreground pixels")

    slices = []
    for dim in range(coords.shape[0]):
        low = max(0, int(coords[dim].min()) - padding)
        high = min(mask.shape[dim], int(coords[dim].max()) + 1 + padding)
        slices.append(slice(low, high))

    return tuple(slices)


def extract_roi(
    image: np.ndarray,
    method: str = "adaptive_threshold",
    return_mask: bool = False,
    min_size: int = 100,
) -> np.ndarray:
    """
    Extract the region of interest from a medical image.

    Pipeline:
        1. Apply adaptive thresholding
        2. Morphological cleanup
        3. Keep largest connected component
        4. Crop to bounding box

    Args:
        image: Input medical image (2D or 3D)
        method: Thresholding method
        return_mask: If True, return mask instead of cropped image
        min_size: Minimum connected component size

    Returns:
        Cropped ROI image (or binary mask if return_mask=True)
    """
    mask = adaptive_threshold(image, method=method)
    mask = morphological_cleanup(mask, min_size=min_size)
    mask = extract_largest_component(mask)

    if return_mask:
        return mask

    bbox = extract_bbox(mask, padding=5)
    return image[bbox]


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: float = 0.0,
) -> np.ndarray:
    """
    Apply a binary mask to an image, setting background pixels.

    Args:
        image: Input image
        mask: Binary mask (same shape as image)
        background_value: Value to set for masked-out regions

    Returns:
        Masked image
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape} vs mask {mask.shape}"
        )

    result = image.copy()
    result[mask == 0] = background_value
    return result


if __name__ == "__main__":
    print("ROI extraction utilities")
    print("Functions available:")
    print("  - adaptive_threshold(image, method)")
    print("  - morphological_cleanup(mask)")
    print("  - extract_largest_component(mask)")
    print("  - extract_bbox(mask, padding)")
    print("  - extract_roi(image, method)")
    print("  - apply_mask(image, mask)")

    # Sanity check
    dummy = np.random.randn(128, 128) + 1.0
    mask = adaptive_threshold(dummy, method="otsu")
    print(f"\nTest mask: {mask.shape}, foreground pixels: {mask.sum()}")
