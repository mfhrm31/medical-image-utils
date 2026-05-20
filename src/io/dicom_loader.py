"""
DICOM loading utilities for medical imaging.

Supports single-file DICOM reads and full CT/MRI series loading
into 3D NumPy volumes with proper voxel spacing.
"""

import numpy as np
import pydicom
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def load_dicom_file(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a single DICOM file.

    Args:
        filepath: Path to .dcm file

    Returns:
        Tuple of:
            - pixel_array: 2D NumPy array of pixel values
            - metadata: Dictionary with key DICOM tags
    """
    ds = pydicom.dcmread(filepath)

    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (for CT HU conversion)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    metadata = {
        "modality": getattr(ds, "Modality", None),
        "patient_id": getattr(ds, "PatientID", None),
        "study_date": getattr(ds, "StudyDate", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "slice_thickness": getattr(ds, "SliceThickness", None),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "bits_stored": int(getattr(ds, "BitsStored", 16)),
    }

    return pixel_array, metadata


def load_dicom_series(folder: str, sort_by_position: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Load a full DICOM series (e.g. CT or MRI volume) into a 3D array.

    Args:
        folder: Path to folder containing .dcm files for one series
        sort_by_position: Whether to sort slices by ImagePositionPatient z-axis

    Returns:
        Tuple of:
            - volume: 3D NumPy array of shape (num_slices, rows, columns)
            - metadata: Dictionary with series-level metadata
    """
    folder_path = Path(folder)
    dcm_files = sorted(folder_path.glob("*.dcm"))

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {folder}")

    datasets = [pydicom.dcmread(str(f)) for f in dcm_files]

    if sort_by_position:
        try:
            datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        except AttributeError:
            datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

    first = datasets[0]
    rows, cols = int(first.Rows), int(first.Columns)
    volume = np.zeros((len(datasets), rows, cols), dtype=np.float32)

    for i, ds in enumerate(datasets):
        slice_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            slice_array = slice_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        volume[i] = slice_array

    # Compute voxel spacing (z, y, x)
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(first, "SliceThickness", 1.0))
    voxel_spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))

    metadata = {
        "modality": getattr(first, "Modality", None),
        "patient_id": getattr(first, "PatientID", None),
        "num_slices": len(datasets),
        "shape": volume.shape,
        "voxel_spacing_mm": voxel_spacing,
        "study_description": getattr(first, "StudyDescription", None),
    }

    return volume, metadata


def list_dicom_series(root_folder: str) -> List[str]:
    """
    Find all subfolders containing DICOM series.

    Args:
        root_folder: Root directory to search

    Returns:
        List of folder paths containing .dcm files
    """
    root = Path(root_folder)
    series_folders = []

    for folder in root.rglob("*"):
        if folder.is_dir() and any(folder.glob("*.dcm")):
            series_folders.append(str(folder))

    return series_folders


if __name__ == "__main__":
    print("DICOM loader utilities")
    print("Functions available:")
    print("  - load_dicom_file(filepath)")
