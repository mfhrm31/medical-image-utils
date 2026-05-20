# medical-image-utils
Lightweight Python toolkit for medical image preprocessing — DICOM loading, intensity normalization, lung/brain segmentation utilities.

**Lightweight Python toolkit for medical image preprocessing**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange)](https://github.com/mfhrm31/medical-image-utils)

A reusable preprocessing toolkit for medical imaging research, supporting common operations on DICOM, NIfTI, and standard image formats used in oncology and radiology workflows.

---

##  Overview

Medical image preprocessing is repetitive across projects: loading DICOM series, windowing CT intensities, normalizing MRI volumes, extracting ROIs. This library consolidates utilities I commonly reuse across my research projects on lung nodule prediction and brain tumor analysis.

Built as a companion library to:
- **[lungnet-hybrid](https://github.com/mfhrm31/lungnet-hybrid)** — Hybrid feature fusion

- medical-image-utils/
├── src/
│   ├── io/                 # DICOM, NIfTI, image loading
│   ├── normalization/      # Intensity normalization and windowing
│   ├── preprocessing/      # ROI extraction, resampling
│   └── augmentation/       # Medical-aware augmentations
├── examples/               # Usage examples on public datasets
├── tests/                  # Unit tests
└── docs/                   # Additional documentation

##  Supported Modalities

| Modality | Format | Status |
|---|---|---|
| CT (Computed Tomography) | DICOM, NIfTI | ✅ Supported |
| MRI (Magnetic Resonance) | DICOM, NIfTI | ✅ Supported |
| X-Ray | PNG, JPEG, DICOM | ✅ Supported |
| Mammography | DICOM | 🚧 In development |
| Ultrasound | DICOM, MP4 | 🚧 Planned |

##  Use Cases

This library is designed to be reused across medical AI research projects. Example use cases:

- **Lung nodule prediction**: HU windowing + adaptive ROI extraction for CT scans
- **Brain tumor MRI**: Skull stripping + intensity normalization across scanners
- **Chest X-ray classification**: Histogram equalization + standardized resizing
- **Multi-site studies**: Cross-scanner intensity harmonization

##  Development Status

This is an active research utility library being built incrementally as I extend my work in medical imaging. Modules marked ✅ are stable and tested; modules marked are in development.

Contributions, suggestions, and bug reports welcome via GitHub Issues.

##  License

MIT License — see [LICENSE](LICENSE) for details.
