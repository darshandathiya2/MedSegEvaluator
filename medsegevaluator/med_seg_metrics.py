# MedicalSegmentationMetrics.py

import numpy as np

class MedicalSegmentationMetrics:
    """
    A simple segmentation metrics class for demonstration.

    **Methods:**
        - dice(y_true, y_pred): Computes the Dice coefficient.
    """

    def dice(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Dice score between two binary masks.

        :param y_true: Ground truth binary mask.
        :param y_pred: Predicted binary mask.
        :return: Dice coefficient (0 to 1).
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2.0 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-8)
