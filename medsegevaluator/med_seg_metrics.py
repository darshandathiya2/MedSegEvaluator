from __future__ import annotations
import numpy as np

__all__ = ["MedicalSegmentationMetrics"]


class MedicalSegmentationMetrics:
    """
    Minimal medical image segmentation evaluation class.

    This version includes only the Dice score for testing and documentation.
    """

    @staticmethod
    def dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Dice score between two binary segmentation masks.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask.
        y_pred : np.ndarray
            Predicted binary mask.

        Returns
        -------
        float
            Dice coefficient ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2.0 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-8)
