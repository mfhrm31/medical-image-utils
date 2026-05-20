"""
Medical-aware data augmentation.

Augmentations specifically designed for medical imaging contexts.
Avoids transformations that would produce clinically implausible
images (e.g. large rotations on chest X-rays, extreme flips on
asymmetric organs).
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class MedicalAugmenter:
    """
    Medical-aware augmentation pipeline.

    Applies random combinations of safe transformations:
        - Small rotations (±15° default)
        - Horizontal flip (only when anatomically appropriate)
        - Intensity shift and scaling (simulates scanner variability)
        - Gaussian noise (simulates acquisition noise)
        - Small translations

    Args:
        rotation_range: Max rotation in degrees (default: 15)
        intensity_shift_range: Max additive shift (default: 0.1)
        intensity_scale_range: Multiplicative scale range (default: 0.1)
        noise_std: Standard deviation of Gaussian noise (default: 0.01)
        translation_range: Max translation as fraction of image size (default: 0.05)
        horizontal_flip_prob: Probability of horizontal flip (default: 0.5)
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        intensity_shift_range: float = 0.1,
        intensity_scale_range: float = 0.1,
        noise_std: float = 0.01,
        translation_range: float = 0.05,
        horizontal_flip_prob: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.rotation_range = rotation_range
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.noise_std = noise_std
        self.translation_range = translation_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rng = np.random.default_rng(random_state)

    def random_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply random small-angle rotation."""
        angle = self.rng.uniform(-self.rotation_range, self.rotation_range)
        return ndimage.rotate(image, angle, reshape=False, order=1, mode="nearest")

    def random_translation(self, image: np.ndarray) -> np.ndarray:
        """Apply random small translation."""
        h, w = image.shape[:2]
        max_dy = int(h * self.translation_range)
        max_dx = int(w * self.translation_range)

        dy = self.rng.integers(-max_dy, max_dy + 1)
        dx = self.rng.integers(-max_dx, max_dx + 1)

        return ndimage.shift(image, [dy, dx], order=1, mode="nearest")

    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Horizontal flip with given probability."""
        if self.rng.random() < self.horizontal_flip_prob:
            return np.fliplr(image)
        return image

    def random_intensity_shift(self, image: np.ndarray) -> np.ndarray:
        """Add random intensity offset (simulates scanner variability)."""
        shift = self.rng.uniform(-self.intensity_shift_range, self.intensity_shift_range)
        return image + shift

    def random_intensity_scale(self, image: np.ndarray) -> np.ndarray:
        """Multiplicative intensity scaling."""
        scale = 1.0 + self.rng.uniform(
            -self.intensity_scale_range, self.intensity_scale_range
        )
        return image * scale

    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate acquisition variability."""
        noise = self.rng.normal(0, self.noise_std, image.shape)
        return image + noise

    def __call__(
        self,
        image: np.ndarray,
        apply_geometric: bool = True,
        apply_intensity: bool = True,
        allow_flip: bool = False,
    ) -> np.ndarray:
        """
        Apply full augmentation pipeline.

        Args:
            image: Input image
            apply_geometric: Whether to apply geometric augmentations
            apply_intensity: Whether to apply intensity augmentations
            allow_flip: Whether horizontal flip is anatomically valid
                        (False for chest X-rays where heart is left-sided)

        Returns:
            Augmented image
        """
        result = image.copy().astype(np.float32)

        if apply_geometric:
            result = self.random_rotation(result)
            result = self.random_translation(result)
            if allow_flip:
                result = self.random_flip(result)

        if apply_intensity:
            result = self.random_intensity_scale(result)
            result = self.random_intensity_shift(result)
            result = self.add_gaussian_noise(result)

        return result


def elastic_deformation(
    image: np.ndarray,
    alpha: float = 1000,
    sigma: float = 30,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Apply elastic deformation, common for medical image augmentation.

    Simulates anatomical variability by applying smooth random displacement
    fields. Particularly effective for organ segmentation training.

    Args:
        image: Input image (2D)
        alpha: Deformation strength
        sigma: Gaussian filter sigma (controls smoothness)
        random_state: Random seed

    Returns:
        Elastically deformed image
    """
    rng = np.random.default_rng(random_state)
    shape = image.shape

    dx = ndimage.gaussian_filter(rng.random(shape) * 2 - 1, sigma) * alpha
    dy = ndimage.gaussian_filter(rng.random(shape) * 2 - 1, sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    deformed = ndimage.map_coordinates(image, indices, order=1, mode="reflect")
    return deformed.reshape(shape)


if __name__ == "__main__":
    print("Medical augmentation utilities")
    augmenter = MedicalAugmenter(random_state=42)
    print(f"\nConfiguration:")
    print(f"  Rotation range:      ±{augmenter.rotation_range}°")
    print(f"  Translation range:   ±{augmenter.translation_range * 100}%")
    print(f"  Intensity shift:     ±{augmenter.intensity_shift_range}")
    print(f"  Intensity scale:     ±{augmenter.intensity_scale_range * 100}%")
    print(f"  Noise std:           {augmenter.noise_std}")
    print(f"  Horizontal flip:     {augmenter.horizontal_flip_prob * 100}%")

    dummy = np.random.randn(128, 128).astype(np.float32)
    augmented = augmenter(dummy, allow_flip=False)
    print(f"\nAugmented image shape: {augmented.shape}")
