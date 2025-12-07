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
    def specificity(y_true: np.ndarray, y_pred: np.ndarray):
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

    @staticmethod
    def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray):
        r"""Compute the symmetric Hausdorff Distance (HD) between two binary masks.
    
        The Hausdorff Distance measures the maximum surface-to-surface distance
        between the predicted and ground-truth segmentation boundaries. It is a
        boundary-level metric commonly used for evaluating segmentation quality in
        medical imaging.
    
        Mathematically, the symmetric Hausdorff Distance is defined as:
    
        .. math::
            HD(A, B) = \max \{ max_{a \in A} d(a, B), max_{b \in B} d(b, A) \}
    
        where :math:`d(a, B) = min_{b \in B} d(a,b)` is the directed Hausdorff distance from set :math:`A`
        to set :math:`B`.
    
        Args:
            y_true (np.ndarray): Ground-truth binary mask.
            y_pred (np.ndarray): Predicted binary mask.
    
        Returns:
            float: Symmetric Hausdorff Distance. If either mask contains no foreground
            pixels, the function returns ``np.inf``.

        Notes:
            This implementation uses ``scipy.spatial.distance.directed_hausdorff``.    
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        y_true_points = np.argwhere(y_true)
        y_pred_points = np.argwhere(y_pred)
    
        # If either mask is empty, HD is undefined → return infinity
        if len(y_true_points) == 0 or len(y_pred_points) == 0:
            return np.inf
    
        d1 = directed_hausdorff(y_true_points, y_pred_points)[0]
        d2 = directed_hausdorff(y_pred_points, y_true_points)[0]
    
        return max(d1, d2)

    @staticmethod
    def hd95(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute the 95th percentile Hausdorff Distance (HD95) between two binary masks.
    
        This metric measures the spatial distance between the boundary points of
        predicted and ground-truth segmentations. HD95 is more stable than the full
        Hausdorff Distance because it ignores extreme outliers.

        Mathematically, the HD95 is defined as:
    
        .. math::
            HD_{95}(A, B) = \max \{  d_{95}(A, B), d_{95}(B, A) \}
    
        where :math:`d_{95}(A, B) = X95_{a \in A} \{ min_{b \in B} d(a,b) \}` is the directed Hausdorff distance from set :math:`A`
        to set :math:`B`.
    
        Args:
            y_true (np.ndarray): Ground truth binary mask. 
            y_pred (np.ndarray): Predicted binary mask. 
            
        Returns:
            float: The 95th percentile Hausdorff distance. Returns ``np.inf`` if one of the masks is empty.
    
        Notes:
            This implementation uses ``scipy.spatial.distance.directed_hausdorff``.
        """
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
    def average_surface_distance(y_true: np.ndarray, y_pred: np.ndarray, voxel_spacing=None):
        r"""
        Compute the Average Surface Distance (ASD) between two binary segmentation masks.
    
        ASD measures the mean symmetric distance between the surfaces (boundaries)
        of two masks. It is computed by averaging the distances from the ground-truth
        surface to the predicted surface **and** vice versa.
    
        .. math::
    
            ASD = \frac{1}{2} \left(
                \frac{1}{|S_{GT}|} \sum_{x \in S_{GT}} d(x, S_{P}) +
                \frac{1}{|S_{P}|} \sum_{y \in S_{P}} d(y, S_{GT})
            \right)
    
        where :math:`S_{GT}` is the ground-truth surface, :math:`S_{P}` is the predicted surface, :math:`d(a, B)` is the minimum Euclidean distance between point, and :math:`a`
              and the set of points :math:`B`.
    
        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask. Non-zero values are treated as foreground.
        y_pred : np.ndarray
            Predicted binary mask. Non-zero values are treated as foreground.
        voxel_spacing : list or tuple of float, optional
            Physical spacing of each voxel dimension (e.g., [x_spacing, y_spacing, z_spacing]).
            Default is isotropic spacing of 1.0.
    
        Returns
        -------
        float
            The average surface distance (ASD). Lower values indicate better boundary alignment.
    
        Notes
        -----
        This function uses:
        - ``scipy.ndimage.distance_transform_edt`` for distance computation
        - ``scipy.ndimage.binary_erosion`` to extract surface voxels
    
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        # handle voxel spacing
        if voxel_spacing is None:
            voxel_spacing = [1.0] * y_true.ndim
    
        # distance transforms of complements
        dt_true = distance_transform_edt(~y_true, sampling=voxel_spacing)
        dt_pred = distance_transform_edt(~y_pred, sampling=voxel_spacing)
    
        # extract surfaces: boundary = foreground AND not eroded foreground
        s_true = np.logical_and(y_true, ~binary_erosion(y_true))
        s_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
    
        # surface distances in each direction
        dist1 = dt_pred[s_true]   # GT surface → Pred surface
        dist2 = dt_true[s_pred]   # Pred surface → GT surface
    
        return (dist1.mean() + dist2.mean()) / 2.0

    @staticmethod
    def nsd(y_true: np.ndarray, y_pred: np.ndarray, tolerance_mm=1.0, voxel_spacing=None):
        r"""
        Compute the **Normalized Surface Dice (NSD)** between two binary masks.
    
        NSD measures how many surface points of both segmentations fall within a
        specified boundary tolerance. It is a widely used metric for clinical
        segmentation challenges because it is robust to small boundary variations.
    
        The metric is defined as:
    
        .. math::
    
            NSD = \frac{
                |\{x \in S_{GT} : d(x, S_{P}) \le \tau \}| +
                |\{y \in S_{P} : d(y, S_{GT}) \le \tau \}|
            }{
                |S_{GT}| + |S_{P}|
            }
    
        where :math:`S_{GT}` is a ground-truth surface, :math:`S_{P}` is a predicted surface, :math:`d(\cdot)` is a Euclidean distance, and :math:`\tau` is a tolerance in millimeters (e.g., 1 mm)
    
        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask. Non-zero values are treated as foreground.
    
        y_pred : np.ndarray
            Predicted binary mask. Non-zero values are treated as foreground.
    
        tolerance_mm : float, optional
            Maximum distance in millimeters for a predicted/ground-truth surface point
            to be considered correctly matched. Default is ``1.0``.
    
        voxel_spacing : float, list, or tuple, optional
            Physical size of voxels in each dimension.
            Examples:
            - ``1.0`` → isotropic 1 mm voxels  
            - ``[0.7, 0.7, 1.0]`` → anisotropic spacing  
            If ``None``, isotropic spacing of 1.0 is used.
    
        Returns
        -------
        float
            Normalized Surface Dice (NSD). Range: ``0.0`` to ``1.0``.
    
        Notes
        -----
        - If both masks are empty → returns ``1.0``.
        - If only one mask is empty → returns ``0.0``.
        - Uses:
          - ``scipy.ndimage.distance_transform_edt`` for point-to-surface distances  
          - ``scipy.ndimage.binary_erosion`` to extract boundary voxels  
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        # handle trivial cases
        if np.all(~y_true) and np.all(~y_pred):
            return 1.0
        if np.all(~y_true) or np.all(~y_pred):
            return 0.0
    
        # handle voxel spacing
        if voxel_spacing is None:
            voxel_spacing = [1.0] * y_true.ndim
        elif np.isscalar(voxel_spacing):
            voxel_spacing = [float(voxel_spacing)] * y_true.ndim
        else:
            voxel_spacing = [float(v) if v is not None else 1.0 for v in voxel_spacing]
    
        # distance transforms
        dt_true = distance_transform_edt(~y_true, sampling=tuple(voxel_spacing))
        dt_pred = distance_transform_edt(~y_pred, sampling=tuple(voxel_spacing))
    
        # extract surfaces
        surf_true = np.logical_and(y_true, ~binary_erosion(y_true))
        surf_pred = np.logical_and(y_pred, ~binary_erosion(y_pred))
    
        # distances
        d_true_to_pred = dt_pred[surf_true]
        d_pred_to_true = dt_true[surf_pred]
    
        # handle no-surface cases
        if len(d_true_to_pred) == 0 or len(d_pred_to_true) == 0:
            return 0.0
    
        # count points within tolerance
        within_true = np.sum(d_true_to_pred <= tolerance_mm)
        within_pred = np.sum(d_pred_to_true <= tolerance_mm)
    
        denom = len(d_true_to_pred) + len(d_pred_to_true)
    
        return (within_true + within_pred) / (denom + 1e-6)
    
    @staticmethod
    def volumetric_similarity(y_true: np.ndarray, y_pred: np.ndarray):
        r"""
        Compute the **Volumetric Similarity (VS)** between two binary segmentation masks.
    
        Volumetric Similarity measures how close the predicted and ground-truth
        volumes are, independent of their spatial alignment. It is particularly useful
        in medical imaging scenarios where anatomical structures may have irregular
        shapes but consistent volumes.
    
        VS is defined as:
    
        .. math::
    
            VS = 1 - \frac{|V_{P} - V_{GT}|}{V_{P} + V_{GT}}
    
        where :math:`V_{P}` is a predicted foreground volume and :math:`V_{GT}` is a ground-truth foreground volume  
    
        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask. Non-zero values are treated as foreground.
    
        y_pred : np.ndarray
            Predicted binary mask. Non-zero values are treated as foreground.
    
        Returns
        -------
        float
            Volumetric similarity between the two masks.
    
        Notes
        -----
        - VS ignores spatial position; it only evaluates volume agreement.
        - Commonly used in medical segmentation challenges (e.g., liver, spleen, brain).
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        v_true = y_true.sum()
        v_pred = y_pred.sum()
    
        return 1 - abs(v_pred - v_true) / (v_pred + v_true + 1e-6)

    
    @staticmethod
    def relative_volume_difference(y_true, y_pred):
        r"""
        Compute the **Relative Volume Difference (RVD)** between two binary segmentation masks.
    
        RVD measures the proportional difference between the predicted and ground-truth
        foreground volumes. A positive value indicates over-segmentation, while a
        negative value indicates under-segmentation.
    
        It is defined as:
    
        .. math::
    
            RVD = \frac{V_{P} - V_{GT}}{V_{GT}}
    
        where :math:`V_{P}` is a predicted foreground volume  and :math:`V_{GT}` is a ground-truth foreground volume  
    
        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary mask. Non-zero values are treated as foreground.
    
        y_pred : np.ndarray
            Predicted binary mask. Non-zero values are treated as foreground.
    
        Returns
        -------
        float
            Relative volume difference between prediction and ground truth.
    
        Notes
        -----
        - RVD is asymmetric: over-segmentation and under-segmentation produce
          values of opposite sign.
        - Widely used in medical image segmentation challenges for volumetric evaluation.
        """
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
    
        v_true = y_true.sum()
        v_pred = y_pred.sum()
    
        return (v_pred - v_true) / (v_true + 1e-6)






























