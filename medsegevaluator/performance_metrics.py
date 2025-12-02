import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff

# ----------------------------- #
# 1. Region-based metrics
# ----------------------------- #

def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-6)

def jaccard_index(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-6)

def precision(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    return tp / (tp + fp + 1e-6)

def recall(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    return tp / (tp + fn + 1e-6)

def specificity(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    return tn / (tn + fp + 1e-6)

def accuracy(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    total = y_true.size
    return (tp + tn) / (total + 1e-6)


# ----------------------------- #
# 2. Surface-based metrics
# ----------------------------- #

def hausdorff_distance(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    y_true_points = np.argwhere(y_true)
    y_pred_points = np.argwhere(y_pred)
    if len(y_true_points) == 0 or len(y_pred_points) == 0:
        return np.inf
    d1 = directed_hausdorff(y_true_points, y_pred_points)[0]
    d2 = directed_hausdorff(y_pred_points, y_true_points)[0]
    return max(d1, d2)

def hd95(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    y_true_points = np.argwhere(y_true)
    y_pred_points = np.argwhere(y_pred)
    if len(y_true_points) == 0 or len(y_pred_points) == 0:
        return np.inf
    d1 = directed_hausdorff(y_true_points, y_pred_points)[0]
    d2 = directed_hausdorff(y_pred_points, y_true_points)[0]
    return np.percentile([d1, d2], 95)

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

def nsd(y_true, y_pred, tolerance_mm=1.0, voxel_spacing=None):
    """Normalized Surface Dice (NSD)"""
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


# ----------------------------- #
# 3. Volume and Overlap metrics
# ----------------------------- #

def volumetric_similarity(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    v_true = y_true.sum()
    v_pred = y_pred.sum()
    return 1 - abs(v_pred - v_true) / (v_pred + v_true + 1e-6)

def relative_volume_difference(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    v_true = y_true.sum()
    v_pred = y_pred.sum()
    return (v_pred - v_true) / (v_true + 1e-6)

def intersection_over_union(y_true, y_pred):
    return jaccard_index(y_true, y_pred)


# ----------------------------- #
# 4. Wrapper for easy evaluation
# ----------------------------- #

def evaluate_all_metrics(y_true, y_pred, voxel_spacing=None):
    return {
        "Dice": dice_score(y_true, y_pred),
        "Jaccard": jaccard_index(y_true, y_pred),
        "Precision": precision(y_true, y_pred),
        "Recall": recall(y_true, y_pred),
        "Specificity": specificity(y_true, y_pred),
        "Accuracy": accuracy(y_true, y_pred),
        "Hausdorff": hausdorff_distance(y_true, y_pred),
        "HD95": hd95(y_true, y_pred),
        "ASD": average_surface_distance(y_true, y_pred, voxel_spacing),
        "NSD": nsd(y_true, y_pred, tolerance_mm=1.0, voxel_spacing=voxel_spacing),
        "Volumetric_Similarity": volumetric_similarity(y_true, y_pred),
        "Relative_Volume_Difference": relative_volume_difference(y_true, y_pred),
        "IoU": intersection_over_union(y_true, y_pred),
    }
