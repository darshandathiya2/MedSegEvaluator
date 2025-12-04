"""
Medical Segmentation Metrics Class
A comprehensive collection of medical image segmentation evaluation metrics.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Tuple, Union, Optional
import warnings

class MedicalSegmentationMetrics:
    """
    Comprehensive medical image segmentation evaluation class.
    
    This class provides all common metrics for evaluating medical image segmentation:
    - Region-based metrics (Dice, Jaccard, Precision, Recall, etc.)
    - Surface-based metrics (Hausdorff distance, ASD, NSD)
    - Volume-based metrics
    - Statistical metrics (TDI, CCC)
    - Robustness metrics
    - Slice-level analysis
    
    Usage Examples:
    ---------------
    # As instance methods (recommended for batch processing):
    evaluator = MedicalSegmentationMetrics(voxel_spacing=[1.0, 1.0, 1.0])
    dice = evaluator.dice_score(gt, pred)
    all_metrics = evaluator.evaluate_all_metrics(gt, pred)
    
    # As static methods (direct function calls):
    dice = MedicalSegmentationMetrics.dice_score_static(gt, pred)
    hd95 = MedicalSegmentationMetrics.hd95_static(gt, pred)
    """
    
    def __init__(self, 
                 voxel_spacing: Optional[Union[List[float], Tuple[float, ...]]] = None,
                 epsilon: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize the metrics evaluator.
        
        Parameters
        ----------
        voxel_spacing : list or tuple, optional
            Physical spacing of voxels in mm (e.g., [1.0, 1.0, 1.0] for isotropic).
        epsilon : float, default=1e-6
            Small value to prevent division by zero.
        verbose : bool, default=False
            Whether to print warnings.
        """
        self.voxel_spacing = self._validate_voxel_spacing(voxel_spacing)
        self.epsilon = epsilon
        self.verbose = verbose
    
    @staticmethod
    def _validate_voxel_spacing(voxel_spacing):
        """Validate and format voxel spacing."""
        if voxel_spacing is None:
            return [1.0]
        elif np.isscalar(voxel_spacing):
            return [float(voxel_spacing)]
        else:
            return [float(v) if v is not None else 1.0 for v in voxel_spacing]
    
    @staticmethod
    def dice_score(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute Dice coefficient (F1 score)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2. * intersection) / (y_true.sum() + y_pred.sum() + epsilon)
    
    @staticmethod
    def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute Jaccard index (Intersection over Union)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / (union + epsilon)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute precision (positive predictive value)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return tp / (tp + fp + epsilon)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute recall (sensitivity, true positive rate)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        return tp / (tp + fn + epsilon)
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute specificity (true negative rate)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tn = np.logical_and(~y_true, ~y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return tn / (tn + fp + epsilon)
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute accuracy."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        tn = np.logical_and(~y_true, ~y_pred).sum()
        total = y_true.size
        return (tp + tn) / (total + epsilon)
    
    @staticmethod
    def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Hausdorff distance."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        y_true_points = np.argwhere(y_true)
        y_pred_points = np.argwhere(y_pred)
        if len(y_true_points) == 0 or len(y_pred_points) == 0:
            return np.inf
        d1 = directed_hausdorff(y_true_points, y_pred_points)[0]
        d2 = directed_hausdorff(y_pred_points, y_true_points)[0]
        return max(d1, d2)
    
    @staticmethod
    def hd95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute 95th percentile Hausdorff distance."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        y_true_points = np.argwhere(y_true)
        y_pred_points = np.argwhere(y_pred)
        if len(y_true_points) == 0 or len(y_pred_points) == 0:
            return np.inf
        d1 = directed_hausdorff(y_true_points, y_pred_points)[0]
        d2 = directed_hausdorff(y_pred_points, y_true_points)[0]
        return np.percentile([d1, d2], 95)
    
    @staticmethod
    def average_surface_distance(y_true: np.ndarray, y_pred: np.ndarray, 
                                       voxel_spacing: Optional[Union[List[float], Tuple[float, ...]]] = None) -> float:
        """Compute average symmetric surface distance."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        
        if voxel_spacing is None:
            voxel_spacing = [1.0] * y_true.ndim
        
        # Ensure voxel_spacing matches dimensions
        if np.isscalar(voxel_spacing):
            voxel_spacing = [float(voxel_spacing)] * y_true.ndim
        else:
            voxel_spacing = [float(v) if v is not None else 1.0 for v in voxel_spacing]
            while len(voxel_spacing) < y_true.ndim:
                voxel_spacing.append(1.0)
            voxel_spacing = voxel_spacing[:y_true.ndim]
        
        dt_true = distance_transform_edt(~y_true, sampling=voxel_spacing)
        dt_pred = distance_transform_edt(~y_pred, sampling=voxel_spacing)
        
        s_true = np.logical_and(y_true, ~binary_erosion(y_true))
        s_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
        
        if s_true.sum() == 0 or s_pred.sum() == 0:
            return np.inf
        
        dist1 = dt_pred[s_true]
        dist2 = dt_true[s_pred]
        
        return (dist1.mean() + dist2.mean()) / 2.0
    
    @staticmethod
    def nsd(y_true: np.ndarray, y_pred: np.ndarray, 
                  tolerance_mm: float = 1.0, 
                  voxel_spacing: Optional[Union[List[float], Tuple[float, ...]]] = None) -> float:
        """Compute Normalized Surface Dice (NSD)."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        
        if np.all(~y_true) and np.all(~y_pred):
            return 1.0
        if np.all(~y_true) or np.all(~y_pred):
            return 0.0
        
        if voxel_spacing is None:
            voxel_spacing = [1.0] * y_true.ndim
        elif np.isscalar(voxel_spacing):
            voxel_spacing = [float(voxel_spacing)] * y_true.ndim
        else:
            voxel_spacing = [float(v) if v is not None else 1.0 for v in voxel_spacing]
        
        dt_true = distance_transform_edt(~y_true, sampling=tuple(voxel_spacing))
        dt_pred = distance_transform_edt(~y_pred, sampling=tuple(voxel_spacing))
        
        surf_true = np.logical_and(y_true, ~binary_erosion(y_true))
        surf_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
        
        d_true_to_pred = dt_pred[surf_true]
        d_pred_to_true = dt_true[surf_pred]
        
        if len(d_true_to_pred) == 0 or len(d_pred_to_true) == 0:
            return 0.0
        
        within_true = np.sum(d_true_to_pred <= tolerance_mm)
        within_pred = np.sum(d_pred_to_true <= tolerance_mm)
        denom = len(d_true_to_pred) + len(d_pred_to_true)
        
        return (within_true + within_pred) / (denom + 1e-6)
    
    @staticmethod
    def volumetric_similarity(y_true: np.ndarray, y_pred: np.ndarray, 
                                    epsilon: float = 1e-6) -> float:
        """Compute volumetric similarity."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        v_true = y_true.sum()
        v_pred = y_pred.sum()
        return 1 - abs(v_pred - v_true) / (v_pred + v_true + epsilon)
    
    @staticmethod
    def relative_volume_difference(y_true: np.ndarray, y_pred: np.ndarray, 
                                         epsilon: float = 1e-6) -> float:
        """Compute relative volume difference."""
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        v_true = y_true.sum()
        v_pred = y_pred.sum()
        return (v_pred - v_true) / (v_true + epsilon)
    
    @staticmethod
    def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Intersection over Union (same as Jaccard index)."""
        return MedicalSegmentationMetrics.jaccard_index(y_true, y_pred)
    
    @staticmethod
    def total_deviation_index(pred: np.ndarray, gt: np.ndarray, p: float = 0.95) -> float:
        """
        Compute Total Deviation Index (TDI_p) between predicted and ground truth masks.
        
        Parameters
        ----------
        pred : np.ndarray
            Predicted mask (can be float [0,1] or binary 0/1)
        gt : np.ndarray
            Ground truth mask (can be float [0,1] or binary 0/1)
        p : float, default=0.95
            Percentile (e.g., 0.95 for TDI95)
        
        Returns
        -------
        float
            TDI value
        """
        # Flatten and compute absolute deviations
        e = np.abs(pred.flatten() - gt.flatten())
        
        # Compute percentile
        tdi = np.percentile(e, p * 100)
        
        return tdi
    
    @staticmethod
    def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray, 
                                                  epsilon: float = 1e-8) -> Tuple[float, float]:
        """
        Compute Concordance Correlation Coefficient (CCC) and Pearson correlation.
        
        Returns
        -------
        tuple
            (CCC, Pearson correlation)
        """
        y_true = np.asarray(y_true).astype(np.float32).flatten()
        y_pred = np.asarray(y_pred).astype(np.float32).flatten()
        
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        
        # Pearson correlation
        rho = cov / (np.sqrt(var_true * var_pred) + epsilon)
        
        # CCC with explicit ρ
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2 + epsilon
        if denominator == 0:
            ccc = 0.0
        else:
            ccc = 2 * cov / denominator
        
        return np.clip(ccc, -1.0, 1.0), rho
    
    @staticmethod
    def dice_delta(original: float, perturbed: float, absolute: bool = False) -> float:
        """
        Compute the change (drop) in Dice score after perturbation.
        
        Parameters
        ----------
        original : float
            Dice score of the original (unperturbed) image.
        perturbed : float
            Dice score under the perturbation.
        absolute : bool, default=False
            If True → return absolute difference |original - perturbed|
            If False → return signed drop (original - perturbed)
        
        Returns
        -------
        float
            Dice drop value.
        """
        drop = original - perturbed
        return abs(drop) if absolute else drop
    
    @staticmethod
    def global_robustness_score(gt: np.ndarray, pred: np.ndarray, 
                                      D_ref: float = 10.0, 
                                      epsilon: float = 1e-6) -> Dict[str, float]:
        """
        Compute combined Global Robustness Score (GRS) integrating Dice, HD95, and CCC.
        
        Parameters
        ----------
        gt : np.ndarray
            Ground truth mask (can be float [0,1] or binary 0/1)
        pred : np.ndarray
            Predicted mask (can be float [0,1] or binary 0/1)
        D_ref : float, default=10.0
            Reference distance for HD95 normalization.
        epsilon : float, default=1e-6
            Small value to prevent division by zero.
        
        Returns
        -------
        dict
            Dictionary with individual metrics and GRS.
        """
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}")
        
        # Compute individual metrics
        dice = MedicalSegmentationMetrics.dice_score_static(gt, pred, epsilon)
        hd = MedicalSegmentationMetrics.hd95(gt, pred)
        ccc, _ = MedicalSegmentationMetrics.concordance_correlation_coefficient(gt, pred, epsilon)
        
        # Normalize HD95 and CCC
        S_H = np.clip(hd / D_ref, 0, 1) if D_ref > 0 else 0
        ccc_norm = np.clip((ccc + 1) / 2.0, 0, 1)
        
        # Compute Global Robustness Score
        grs = (dice + (1 - S_H) + ccc_norm) / 3.0
        
        return {
            "Dice": dice,
            "HD95": hd,
            "CCC": ccc,
            "CCC_Norm": ccc_norm,
            "S_H": S_H,
            "GRS": grs
        }
    
    @staticmethod
    def dice_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-8) -> float:
        """Compute Dice from counts (numerically stable)."""
        return (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    
    @staticmethod
    def slice_level_dice(gt3d: np.ndarray, pred3d: np.ndarray,
                               slice_axis: int = 0,
                               ignore_empty_slices: bool = True,
                               empty_slice_value: float = 1.0,
                               smooth: float = 1e-6) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute per-slice Dice for 3D binary masks and return summary statistics.
        
        Parameters
        ----------
        gt3d, pred3d : np.ndarray
            Ground truth & predicted masks (same shape).
        slice_axis : int, default=0
            Along which dimension to compute slice-level Dice.
        ignore_empty_slices : bool, default=True
            Whether to completely skip empty-empty slices.
        empty_slice_value : float, default=1.0
            Value assigned when GT and Pred both empty (if not ignored).
        smooth : float, default=1e-6
            Smoothing factor to prevent division by zero.
        
        Returns
        -------
        tuple
            (dice_per_slice_array, summary_statistics_dict)
        """
        if gt3d.shape != pred3d.shape:
            raise ValueError("gt3d and pred3d must have same shape")
        
        # Move slicing axis to index 0
        if slice_axis != 0:
            gt3d = np.moveaxis(gt3d, slice_axis, 0)
            pred3d = np.moveaxis(pred3d, slice_axis, 0)
        
        num_slices = gt3d.shape[0]
        dices = np.zeros(num_slices, dtype=float)
        empty_mask = np.zeros(num_slices, dtype=bool)
        
        for i in range(num_slices):
            g = (gt3d[i] > 0).astype(np.uint8)
            p = (pred3d[i] > 0).astype(np.uint8)
            
            g_empty = g.sum() == 0
            p_empty = p.sum() == 0
            
            # Condition A — ignore empty slices
            if ignore_empty_slices:
                if g_empty and p_empty:
                    empty_mask[i] = True
                    dices[i] = np.nan
                    continue
                elif g_empty and not p_empty:
                    dices[i] = 0.0
                    continue
            
            # Condition B — formal Dice including empty cases
            if not ignore_empty_slices:
                if g_empty and p_empty:
                    dices[i] = empty_slice_value
                    empty_mask[i] = True
                    continue
                elif g_empty and not p_empty:
                    dices[i] = 0.0
                    continue
                elif not g_empty and p_empty:
                    dices[i] = 0.0
                    continue
            
            # Normal Dice formula
            intersection = np.sum(g * p)
            dices[i] = (2 * intersection + smooth) / (g.sum() + p.sum() + smooth)
        
        # Summary statistics
        nonempty_mask = ~empty_mask
        num_empty = int(empty_mask.sum())
        num_nonempty = int(nonempty_mask.sum())
        
        if num_nonempty > 0:
            mean_nonempty = float(np.nanmean(dices[nonempty_mask]))
            proportion_below_0_9 = float((dices[nonempty_mask] < 0.9).sum() / num_nonempty)
            proportion_below_0_8 = float((dices[nonempty_mask] < 0.8).sum() / num_nonempty)
        else:
            mean_nonempty = np.nan
            proportion_below_0_9 = np.nan
            proportion_below_0_8 = np.nan
        
        mean_all = float(np.nanmean(dices))
        
        stats = {
            "num_slices": num_slices,
            "num_empty_slices": num_empty,
            "num_nonempty_slices": num_nonempty,
            "mean_all": mean_all,
            "mean_nonempty": mean_nonempty,
            "proportion_below_0.9": proportion_below_0_9,
            "proportion_below_0.8": proportion_below_0_8,
        }
        
        return dices, stats
    
    @staticmethod
    def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   voxel_spacing: Optional[Union[List[float], Tuple[float, ...]]] = None) -> Dict[str, float]:
        """
        Evaluate all metrics at once.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth mask.
        y_pred : np.ndarray
            Predicted mask.
        voxel_spacing : list or tuple, optional
            Voxel spacing for surface metrics.
        
        Returns
        -------
        dict
            Dictionary containing all computed metrics.
        """
        return {
            "Dice": MedicalSegmentationMetrics.dice_score(y_true, y_pred),
            "Jaccard": MedicalSegmentationMetrics.jaccard_index(y_true, y_pred),
            "Precision": MedicalSegmentationMetrics.precision(y_true, y_pred),
            "Recall": MedicalSegmentationMetrics.recall(y_true, y_pred),
            "Specificity": MedicalSegmentationMetrics.specificity(y_true, y_pred),
            "Accuracy": MedicalSegmentationMetrics.accuracy(y_true, y_pred),
            "Hausdorff": MedicalSegmentationMetrics.hausdorff_distance(y_true, y_pred),
            "HD95": MedicalSegmentationMetrics.hd95(y_true, y_pred),
            "ASD": MedicalSegmentationMetrics.average_surface_distance(y_true, y_pred, voxel_spacing),
            "NSD": MedicalSegmentationMetrics.nsd(y_true, y_pred, tolerance_mm=1.0, voxel_spacing=voxel_spacing),
            "Volumetric_Similarity": MedicalSegmentationMetrics.volumetric_similarity(y_true, y_pred),
            "Relative_Volume_Difference": MedicalSegmentationMetrics.relative_volume_difference(y_true, y_pred),
            "IoU": MedicalSegmentationMetrics.intersection_over_union(y_true, y_pred),
        }
    
    # =========================================================================
    # ADDITIONAL UTILITY METHODS
    # =========================================================================
    
    def print_summary( y_true: np.ndarray, y_pred: np.ndarray):
        """Print a comprehensive summary of all metrics."""
        metrics = MedicalSegmentationMetrics.evaluate_all_metrics(y_true, y_pred)
        
        print("=" * 70)
        print("MEDICAL SEGMENTATION METRICS SUMMARY")
        print("=" * 70)
        
        print("\n1. REGION-BASED METRICS:")
        print("-" * 40)
        for metric in ["Dice", "Jaccard", "Precision", "Recall", "Specificity", "Accuracy"]:
            print(f"  {metric:25s}: {metrics.get(metric, 'N/A'):.4f}")
        
        print("\n2. SURFACE-BASED METRICS:")
        print("-" * 40)
        for metric in ["Hausdorff", "HD95", "ASD", "NSD"]:
            value = metrics.get(metric, np.inf)
            if np.isfinite(value):
                print(f"  {metric:25s}: {value:.4f}")
            else:
                print(f"  {metric:25s}: INF")
        
        print("\n3. VOLUME-BASED METRICS:")
        print("-" * 40)
        for metric in ["Volumetric_Similarity", "Relative_Volume_Difference", "IoU"]:
            print(f"  {metric:25s}: {metrics.get(metric, 'N/A'):.4f}")
        
        print("=" * 70)
    
    def batch_evaluate(y_true_list: List[np.ndarray], 
                      y_pred_list: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Evaluate multiple image pairs.
        
        Parameters
        ----------
        y_true_list : list of np.ndarray
            List of ground truth masks.
        y_pred_list : list of np.ndarray
            List of predicted masks.
        
        Returns
        -------
        list
            List of metric dictionaries.
        """
        if len(y_true_list) != len(y_pred_list):
            raise ValueError(f"Length mismatch: {len(y_true_list)} ground truth vs {len(y_pred_list)} predictions")
        
        results = []
        for gt, pred in zip(y_true_list, y_pred_list):
            results.append(MedicalSegmentationMetrics.evaluate_all_metrics(gt, pred))
        
        return results



