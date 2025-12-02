Performance Metrics
===================

MedSegEvaluator provides a wide range of segmentation metrics grouped into
four categories: region-level overlap, boundary accuracy, surface distances,
and volumetric agreement.

.. automodule:: medsegevaluator.PerformanceMetrics
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

The :class:`PerformanceMetrics` class provides a unified collection of
region-based, surface-based, volumetric, and robustness metrics commonly used
in medical image segmentation evaluation.

All metrics are implemented as ``@staticmethod`` because they do not depend on
class instance state.

This section summarizes the four major categories of segmentation metrics
available in MedSegEvaluator.

- **Region-Level Metrics**  
  Measures overlap quality between prediction and ground truth masks.

- **Boundary-Level Metrics**  
  Measures how well predicted boundaries align with the true anatomical contour.

- **Surface Distances**  
  Measures geometric distances between the surfaces of two segmentations.

- **Volumetric Agreement**  
  Measures how similar the region volumes are between prediction and ground truth.

Each category captures different aspects of segmentation performance.

.. contents::
   
   :local:
   :depth: 2



====================================
Region-Level Metrics
====================================

It quantifies the **overlap** between the predicted and ground truth
masks.

Accuracy
--------

Accuracy measures the overall proportion of correctly classified pixels in the
segmentation mask. Although commonly used in classification tasks, it should be
interpreted cautiously for medical image segmentation, especially in the presence
of class imbalance.

Formula:

.. math::

    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

Where :math:`TP`, :math:`TN`, :math:`FP`, and :math:`FN` are True Positives, True Negatives, False Positives, and False Negatives  
respectively.

Usage::

    from performance_metrics import accuracy
    acc = accuracy(gt, pred)


Precision
----------

A Precision is a proportion of all accurately predicted positive instances among all positive instances. It measures the model's ability.

Formula:

.. math::

   \text{Precision} = \frac{TP}{TP + FP}

Where :math:`TP` and :math:`FP` is  True Positives and False Negatives, respectively.

Usage::

   from performance_metrics import precision
   prec = precision(gt, pred)

Recall
------

A recall is a proportion of correctly predicted positive instances to the total actual positive instances. It measures the model's ability to correctly identify all true positive instances from the total number of actual positive cases.

Formula:

.. math::

  \text{Recall} = \frac{TP}{TP + FN}

Where :math:`TP` and :math:`FN` is  True Positives and False Negatives, respectively.

Usage::

  from performance_metrics import recall
  rc = recall(gt, pred)

Dice Similarity Coefficient
---------------------------

The Dice coefficient measures the overlap between the predicted segmentation and the ground truth.

Formula:

.. math::

    \text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}

Where :math:`A` is the set of voxels/pixels in the ground truth mask, :math:`B` is the set of voxels/pixels in the predicted mask.

Usage::

    from performance_metrics import dice_score
    score = dice_score(gt, pred)


Inetersection over Union (IOU)
------------------------------

The Intersection over Union (IoU), also known as the Jaccard Index, measures the ratio between the area of overlap and the area of union of the predicted segmentation and the ground truth.

Formula:

.. math::
  \text{IoU} = \frac{|A \cap B|}{|A \cup B|}

Where :math:`A` is the set of voxels/pixels in the ground truth mask, :math:`B` is the set of voxels/pixels in the predicted mask.

Usage::

  from performance_metrics import intersection_over_union
  iou = intersection_over_union(gt, pred)


====================================
Boundary-level Metrics
====================================

Boundary-level metrics measure the accuracy of predicted contours against anatomical boundaries.

Housedroff Distance (HD)
-------------------

The Hausdorff Distance (HD) measures the maximum boundary deviation between the predicted segmentation and the ground truth. 

Formula:

.. math::

  \text{HD}(X, Y) = \max \left\{ \sup_{x \in X} \inf_{y \in Y} d(x, y), \sup_{y \in Y} \inf_{x \in X} d(x, y) \right\}. 

Usage::

  from performance_metrics import hausdorff_distance
  HD = hausdorff_distance(gt, pred)


95th Percentile Hausdorff Distance (HD95)
-----------------------------------------

The 95th Percentile Hausdorff Distance (HD95) is a robust version of the Hausdorff Distance. Instead of the maximum surface deviation, it computes the 95th percentile of all boundary distances.

Formula:

.. math::

    HD_{95}(A, B) = \max \big( P_{95}\big(d(A, B)\big), \, P_{95}\big(d(B, A)\big) \big),

where :math:`P_{95}(\cdot)` denotes the 95th percentile operator and :math:`d(A, B)` be all distances from prediction boundary :math:`A` to ground truth :math:`B`, and :math:`d(B, A)` is vice-versa.

Usage::

  from performance_metrics import hd95
  HD95 = hd95(gt, pred)


Average Symmetric Surface Distance (ASSD)
-----------------------------------------

The Average Symmetric Surface Distance (ASSD) is a metric used to measure the average distance between the boundaries of two segmentations, such as a machine-segmented region and a ground truth. It is calculated by finding the average of all distances from every point on the first surface to the closest point on the second surface, and then averaging this with the distances from every point on the second surface to the closest point on the first.

Usage::

  from performance_metrics import average_surface_distance
  assd = average_surface_distance(gt, pred)
  



