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

Accuracy
--------

Accuracy measures the overall proportion of correctly classified pixels in the
segmentation mask. Although commonly used in classification tasks, it should be
interpreted cautiously for medical image segmentation, especially in the presence
of class imbalance.

Formula:

.. math::

    Accuracy = \frac{TP + TN}{TP + TN + FP + FN}

Where :math:`TP`,:math:`TN`,  :math:`FP`, and :math:`FN` are True Positives, True Negatives, False Positives, and False Negatives  
respectively.

Usage::

    from medsegevaluator.metrics import accuracy
    score = accuracy(gt, pred)


Precision
----------

A Precision is a proportion of all accurately predicted positive instances among all positive instances. It measures the model's ability.

Formula:

.. math::

   Precision = \frac{TP}{TP + FP}

Where :math:`TP` and :math:`FP` is  True Positives and False Negatives, respectively.

Usage::

   from performance_metrics import precision
   prec = precision(gt, pred)

Recall
------

A recall is a proportion of correctly predicted positive instances to the total actual positive instances. It measures the model's ability to correctly identify all true positive instances from the total number of actual positive cases.

Formula:

.. math::

  Recall = \frac{TP}{TP + FN}

Where :math:`TP` and :math:`FN` is  True Positives and False Negatives, respectively.


Dice Similarity Coefficient
---------------------------

The Dice coefficient measures the overlap between ground truth and 
predicted segmentation.

Formula:

.. math::

    Dice = \frac{2 \cdot |A \cap B|}{|A| + |B|}

Where :math:`A` is the set of voxels/pixels in the ground truth mask, :math:`B` is the set of voxels/pixels in the predicted mask.

Usage::

    from performance_metrics import dice_score
    score = dice_score(gt, pred)


Usage::

  from performance_metrics import recall
  rc = recall(gt, pred)
