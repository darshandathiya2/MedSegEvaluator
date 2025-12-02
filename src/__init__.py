"""
medSegEvaluator
================

A Python library for evaluating medical image segmentation models.

Modules:
--------
- bland_altman_plot        : Bland–Altman visualization for segmentation agreement.
- image_morphology         : Compute volume, area, and island count of segmented regions.
- image_quality            : Compute image quality metrics (SNR, Entropy, Blur).
- medicalimageloader       : Load medical images (e.g., NIfTI,DICOM, PNG) as normalized NumPy arrays.
- performance_visualization: Plot distributions, boxplots, and comparison charts.
- performance_metrics      : Compute Dice, IoU, Hausdorff, ASD, and related metrics.
"""

from .bland_altman_plot import bland_altman_plot
from .image_morphology import get_volume, get_island_count, get_area
from .image_quality import compute_snr, compute_entropy, compute_blur, evaluate_volume_from_array
from .medicalimageloader import load_image
from .performance_visualization import (
    plot_histogram_distribution,
    plot_dice_vs_islands_flexible
)
from .performance_metrics import (
    evaluate_all_metrics,
    dice_score,
    iou_score,
    hausdorff_distance,
    average_surface_distance
)

__all__ = [
    "bland_altman_plot",
    "get_volume",
    "get_island_count",
    "get_area",
    "compute_snr",
    "compute_entropy",
    "compute_blur",
    "evaluate_volume_from_array",
    "load_image",
    "plot_histogram_distribution",
    "plot_dice_vs_islands_flexible",
    "evaluate_all_metrics",
    "dice_score",
    "iou_score",
    "hausdorff_distance",
    "average_surface_distance"
]
