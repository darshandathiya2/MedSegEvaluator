Performance Metrics
===================

MedSegEvaluator provides a comprehensive suite of metrics for assessing 
medical image segmentation quality. These metrics cover region-level 
overlap, boundary accuracy, surface distances, and volumetric agreement.


This page describes each metric, its mathematical formulation, and 
how to compute it using MedSegEvaluator.

Overview
--------

The available segmentation metrics include:

- **Region-Level Overlap**

   - Dice Similarity Coefficient
   - Intersection over Union (IoU)
   - Precision & Recall
   - F1 Score

- **Boundary Accuracy**

   - Hausdorff Distance (HD95)

- **Surface Distances**

   - Average Symmetric Surface Distance (ASSD)
   - Surface Dice

- **Volumetric Agreement**

   - Volume Similarity (VS)

Content
-------

.. contents::
   :local:
   :depth: 2


Dice Similarity Coefficient
---------------------------

The Dice coefficient measures the overlap between ground truth and 
predicted segmentation.

Formula:

.. math::

    Dice = \frac{2 \cdot |A \cap B|}{|A| + |B|}

Where: 
``A`` is the set of voxels/pixels in the ground truth mask, ``B`` is the set of voxels/pixels in the predicted mask.

Usage::

    from performance_metrics import dice_score
    score = dice_score(gt, pred)
