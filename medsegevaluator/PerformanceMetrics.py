from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff


class PerformanceMetrics:
    r"""
    Class containing a comprehensive set of segmentation evaluation metrics.
    
    Metrics include:
        1. Region-based: Dice, Jaccard, Precision, Recall, Specificity, Accuracy
        2. Surface-based: Hausdorff Distance, HD95, Average Surface Distance
        3. Volume-based: Volumetric Similarity, Relative Volume Difference, IoU
        4. Robustness: Dice Drop, Global Robustness Score
        5. Slice-level (3D): Slice-wise Dice and statistics
        6. Utility: Concordance Correlation Coefficient (CCC)
    """

    # ------------------------------------------------------------- #
    # 1. REGION-BASED METRICS
    # ------------------------------------------------------------- #
    @staticmethod
    def dice_score(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-6)

    @staticmethod
    def jaccard_index(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / (union + 1e-6)

    @staticmethod
    def precision(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return tp / (tp + fp + 1e-6)

    @staticmethod
    def recall(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        return tp / (tp + fn + 1e-6)

    @staticmethod
    def specificity(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tn = np.logical_and(~y_true, ~y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return tn / (tn + fp + 1e-6)

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        tp = np.logical_and(y_true, y_pred).sum()
        tn = np.logical_and(~y_true, ~y_pred).sum()
        total = y_true.size
        return (tp + tn) / (total + 1e-6)

    # ------------------------------------------------------------- #
    # 2. SURFACE-BASED METRICS
    # ------------------------------------------------------------- #
    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        p_true = np.argwhere(y_true)
        p_pred = np.argwhere(y_pred)
        if len(p_true) == 0 or len(p_pred) == 0:
            return np.inf
        d1 = directed_hausdorff(p_true, p_pred)[0]
        d2 = directed_hausdorff(p_pred, p_true)[0]
        return max(d1, d2)

    @staticmethod
    def hd95(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        p_true = np.argwhere(y_true)
        p_pred = np.argwhere(y_pred)
        if len(p_true) == 0 or len(p_pred) == 0:
            return np.inf
        d1 = directed_hausdorff(p_true, p_pred)[0]
        d2 = directed_hausdorff(p_pred, p_true)[0]
        return np.percentile([d1, d2], 95)

    @staticmethod
    def average_surface_distance(y_true, y_pred, voxel_spacing=None):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        if voxel_spacing is None:
            voxel_spacing = [1.0] * y_true.ndim

        dt_true = distance_transform_edt(~y_true, sampling=voxel_spacing)
        dt_pred = distance_transform_edt(~y_pred, sampling=voxel_spacing)

        s_true = np.logical_and(y_true, ~binary_erosion(y_true))
        s_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))

        dist1 = dt_pred[s_true]
        dist2 = dt_true[s_pred]

        return (dist1.mean() + dist2.mean()) / 2.0

    # ------------------------------------------------------------- #
    # 3. VOLUME-BASED METRICS
    # ------------------------------------------------------------- #
    @staticmethod
    def volumetric_similarity(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        v_true = y_true.sum()
        v_pred = y_pred.sum()
        return 1 - abs(v_pred - v_true) / (v_pred + v_true + 1e-6)

    @staticmethod
    def relative_volume_difference(y_true, y_pred):
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        v_true = y_true.sum()
        v_pred = y_pred.sum()
        return (v_pred - v_true) / (v_true + 1e-6)

    @staticmethod
    def intersection_over_union(y_true, y_pred):
        return PerformanceMetrics.jaccard_index(y_true, y_pred)

    # ------------------------------------------------------------- #
    # 4. ROBUSTNESS METRICS
    # ------------------------------------------------------------- #
    @staticmethod
    def dice_drop(original, perturbed, absolute=False):
        drop = original - perturbed
        return abs(drop) if absolute else drop

    @staticmethod
    def global_robustness_score(gt, pred, D_ref=10.0):
        dice = PerformanceMetrics.dice_score(gt, pred)
        hd = PerformanceMetrics.hd95(gt, pred)
        ccc = PerformanceMetrics.concordance_correlation_coefficient(gt, pred)[0]

        S_H = np.clip(hd / D_ref, 0, 1)
        ccc_norm = np.clip((ccc + 1) / 2.0, 0, 1)

        grs = (dice + (1 - S_H) + ccc_norm) / 3.0

        return {
            "Dice": dice,
            "HD95": hd,
            "CCC": ccc,
            "CCC_Norm": ccc_norm,
            "S_H": S_H,
            "GRS": grs
        }

    # ------------------------------------------------------------- #
    # 5. SLICE-LEVEL METRICS (3D)
    # ------------------------------------------------------------- #
    @staticmethod
    def slice_level_dice(gt3d, pred3d, slice_axis=0,
                         ignore_empty_slices=True,
                         empty_slice_value=1.0,
                         smooth=1e-6):
        if gt3d.shape != pred3d.shape:
            raise ValueError("gt3d and pred3d must have same shape")

        if slice_axis != 0:
            gt3d = np.moveaxis(gt3d, slice_axis, 0)
            pred3d = np.moveaxis(pred3d, slice_axis, 0)

        num_slices = gt3d.shape[0]
        dices = np.zeros(num_slices, float)
        empty_mask = np.zeros(num_slices, bool)

        for i in range(num_slices):
            g = (gt3d[i] > 0).astype(np.uint8)
            p = (pred3d[i] > 0).astype(np.uint8)

            g_empty = g.sum() == 0
            p_empty = p.sum() == 0

            if ignore_empty_slices:
                if g_empty and p_empty:
                    empty_mask[i] = True
                    dices[i] = np.nan
                    continue
                elif g_empty and not p_empty:
                    dices[i] = 0.0
                    continue

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

            intersection = np.sum(g * p)
            dices[i] = (2 * intersection + smooth) / (g.sum() + p.sum() + smooth)

        nonempty = ~empty_mask
        num_empty = empty_mask.sum()
        num_nonempty = nonempty.sum()

        if num_nonempty > 0:
            mean_nonempty = float(np.nanmean(dices[nonempty]))
            below_0_9 = float((dices[nonempty] < 0.9).sum() / num_nonempty)
            below_0_8 = float((dices[nonempty] < 0.8).sum() / num_nonempty)
        else:
            mean_nonempty = np.nan
            below_0_9 = np.nan
            below_0_8 = np.nan

        stats = {
            "num_slices": num_slices,
            "num_empty_slices": int(num_empty),
            "num_nonempty_slices": int(num_nonempty),
            "mean_all": float(np.nanmean(dices)),
            "mean_nonempty": mean_nonempty,
            "proportion_below_0.9": below_0_9,
            "proportion_below_0.8": below_0_8,
        }

        return dices, stats

    # ------------------------------------------------------------- #
    # 6. UTILITY METRICS
    # ------------------------------------------------------------- #
    @staticmethod
    def concordance_correlation_coefficient(y_true, y_pred, epsilon=1e-8):
        y_true = np.asarray(y_true).astype(np.float32).flatten()
        y_pred = np.asarray(y_pred).astype(np.float32).flatten()
        m1, m2 = y_true.mean(), y_pred.mean()
        v1, v2 = y_true.var(), y_pred.var()
        cov = np.mean((y_true - m1) * (y_pred - m2))
        rho = cov / (np.sqrt(v1 * v2) + epsilon)
        ccc = rho * (2 * np.sqrt(v1 * v2)) / (v1 + v2 + (m1 - m2)**2 + epsilon)
        return np.clip(ccc, -1, 1), rho

    # ------------------------------------------------------------- #
    # 7. WRAPPER (ALL METRICS)
    # ------------------------------------------------------------- #
    @staticmethod
    def evaluate_all_metrics(y_true, y_pred, voxel_spacing=None):
        return {
            "Dice": PerformanceMetrics.dice_score(y_true, y_pred),
            "Jaccard": PerformanceMetrics.jaccard_index(y_true, y_pred),
            "Precision": PerformanceMetrics.precision(y_true, y_pred),
            "Recall": PerformanceMetrics.recall(y_true, y_pred),
            "Specificity": PerformanceMetrics.specificity(y_true, y_pred),
            "Accuracy": PerformanceMetrics.accuracy(y_true, y_pred),
            "Hausdorff": PerformanceMetrics.hausdorff_distance(y_true, y_pred),
            "HD95": PerformanceMetrics.hd95(y_true, y_pred),
            "ASD": PerformanceMetrics.average_surface_distance(y_true, y_pred, voxel_spacing),
            "Volumetric_Similarity": PerformanceMetrics.volumetric_similarity(y_true, y_pred),
            "Relative_Volume_Difference": PerformanceMetrics.relative_volume_difference(y_true, y_pred),
            "IoU": PerformanceMetrics.intersection_over_union(y_true, y_pred),
            "CCC": PerformanceMetrics.concordance_correlation_coefficient(y_true, y_pred)[0]
        }
