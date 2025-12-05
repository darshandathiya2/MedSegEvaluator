from __future__ import annotations
import numpy as np

__all__ = ["MedicalSegmentationMetrics"]


class MedicalSegmentationMetrics:

    @staticmethod
    def dice(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute Dice score between two binary segmentation masks.
        
        .. math::
          \text{Dice Coefficient} = \frac{2 \cdot |A \cap B|}{|A| + |B|}

        where :math:`A` denotes the predicted set of pixels and :math:`B` denotes the set of ground truth pixels.

        Args:
            y_true : np.ndarray
                Ground-truth binary mask.
            y_pred : np.ndarray
                Predicted binary mask.

        Returns
        -------
        float
            Dice coefficient ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-6)


    @staticmethod
    def iou(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute Intersection over Union (IoU) between two binary segmentation masks.
        
        .. math::
          \text{IoU} = \frac{|A \cap B|}{|A \cup B|}
   
        where :math:`A` denotes the predicted set of pixels and :math:`B` denotes the set of ground truth pixels.

        Args:
            y_true : np.ndarray
                Ground-truth binary mask.
            y_pred : np.ndarray
                Predicted binary mask.

        Returns
        -------
        float
            IoU score ranging from 0 (no overlap) to 1 (perfect overlap).
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / (union + 1e-6)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute classification accuracy between two binary segmentation masks.
    
        Accuracy measures the proportion of correctly classified pixels, including
        both foreground and background.
    
        .. math::
            \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    
        where :math:`TP`, :math:`TN`, :math:`FP`, and :math:`FN` are  true positives, true negatives, false positives, and false negatives respectively. 
    
        Although accuracy is intuitive, it may be misleading in highly imbalanced medical images where the background dominates.
        
        Args:
            y_true (np.ndarray): Ground-truth binary mask.
            y_pred (np.ndarray): Predicted binary segmentation mask.
            
        Returns
        -------
        float
            Accuracy score ranging from 0 (completely incorrect) to 1 (perfect match).
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        tn = np.logical_and(~y_true, ~y_pred).sum()
        total = y_true.size
        return (tp + tn) / (total + 1e-6)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute the Precision score for binary segmentation masks.
    
        Precision measures the proportion of predicted positive pixels that are
        correctly identified.
    
        .. math::
            \text{Precision} = \frac{TP}{TP + FP}
    
        Args:
            y_true (np.ndarray): Ground-truth binary mask.
            y_pred (np.ndarray): Predicted binary segmentation mask.
    
        Returns:
            float: Precision score in the range [0, 1], where higher values indicate fewer false positives.
        """

        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
    
        return tp / (tp + fp + 1e-6)

    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute the Recall score for binary segmentation masks.
    
        Recall measures the proportion of ground-truth positive pixels that are
        correctly detected by the model.
    
        .. math::
            \text{Recall} = \frac{TP}{TP + FN}
    
        where :math:`TP` is the number of true positive pixels and :math:`FN` is the number of false negative pixels.
    
        Args:
            y_true (np.ndarray): Ground-truth binary mask.
            y_pred (np.ndarray): Predicted binary segmentation mask.
            
        Returns:
            float: Recall score in the range [0, 1], where higher values indicate fewer false negatives.
        """
        
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        
        return tp / (tp + fn + 1e-6)

    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r"""Compute the Specificity score for binary segmentation masks.
    
        Specificity measures how well the model identifies background pixels
        correctly. It is defined as:
    
        .. math::
             \text{Specificity} = \frac{TN}{TN + FP}
    
        where :math:`TN` (true negatives) are background pixels correctly predicted and :math:`FP` (false positives) are background pixels 
        incorrectly predicted as foreground.
    
        Args:
            y_true (np.ndarray): Ground-truth binary mask.
            y_pred (np.ndarray): Predicted binary mask.
    
        Returns:
            float: Specificity score ranging from 0 to 1. Higher values indicate
            fewer false positives and better background classification.
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        tn = np.logical_and(~y_true, ~y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
    
        return tn / (tn + fp + 1e-6)




















