Performance Metrics
===================

MedSegEvaluator provides a wide range of segmentation metrics grouped into
four categories: region-level overlap, boundary accuracy, surface distances,
and volumetric agreement.

This section summarizes the four major categories of segmentation metrics
available in MedSegEvaluator.

- **Region-Level Overlap**  
  Measures overlap quality between prediction and ground truth masks.

- **Boundary Accuracy**  
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
Region-Level Overlap
====================================

Metrics that quantify **overlap agreement** between the predicted and ground truth
masks.


Dice Similarity Coefficient
---------------------------

The Dice coefficient measures the overlap between ground truth and 
predicted segmentation.

Formula:

.. math::

    Dice = \frac{2 \cdot |A \cap B|}{|A| + |B|}

Where: 
:math:`A` is the set of voxels/pixels in the ground truth mask, :math:`B` is the set of voxels/pixels in the predicted mask.

Usage::

    from performance_metrics import dice_score
    score = dice_score(gt, pred)


Precision
----------

A Precision is a proportion of all accurately predicted positive instances among all positive instances. It measures the model's ability.

Formula:

.. math::

   Precision = \frac{TP}{TP + FP}

Where: :math:`TP` and :math:`FP` is  True Positives and False Negatives, respectively.

Usage::

   from performance_metrics import precision
   pres = precision(gt, pred)

