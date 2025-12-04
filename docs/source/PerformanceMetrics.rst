Medical Image Segmentation Metrics
============================

The ``PerformanceMetrics`` module provides a comprehensive collection of medical image segmentation evaluation metrics.

This class implements both region-based and surface-based metrics commonly used in medical image analysis, including Dice coefficient, Jaccard index, Hausdorff distance, and more.

Class Reference
---------------

.. autoclass:: medsegevaluator.PerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Summary
-------

.. autosummary::
   :toctree: generated/
   :recursive:
   
   PerformanceMetrics
   PerformanceMetrics.__init__
   PerformanceMetrics.dice_score
   PerformanceMetrics.jaccard_index
   PerformanceMetrics.hausdorff_distance
   PerformanceMetrics.evaluate_all_metrics
