from __future__ import annotations
import numpy as np

__all__ = ["MedicalSegmentationMetrics"]


class MedicalSegmentationMetrics:

    @staticmethod
    def dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Compute Dice score between two binary segmentation masks.
        
        .. math::
          \text{Dice Coefficient} = \frac{2 \cdot |A \cap B|}{|A| + |B|}

        where :math:`A` denotes the predicted set of pixels and :math:`B` denotes the set of ground truth pixels.

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


    @staticmethod
    def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two binary segmentation masks.

        .. math::
          \text{IoU} = \frac{|A \cap B|}{|A \cup B|}

          where :math:`A` denotes the predicted set of pixels and :math:`B` denotes the set of ground truth pixels.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask.
        y_pred : np.ndarray
            Predicted binary mask.

        Returns
        -------
        float
            IoU score ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return intersection / (union + 1e-8)







