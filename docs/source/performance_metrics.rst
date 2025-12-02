Performance Metrics
==================

.. currentmodule:: medsegevaluator.PerformanceMetrics

Class: PerformanceMetrics
-------------------------

.. autoclass:: PerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :noindex:


Overview
--------

The :class:`PerformanceMetrics` class provides a comprehensive set of evaluation
methods for medical image segmentation. It includes region-based, surface-based,
volume-based, robustness, slice-level (3D), and utility metrics.

Example
-------

.. code-block:: python

    from medsegevaluator.PerformanceMetrics import PerformanceMetrics

    dice = PerformanceMetrics.dice_score(y_true, y_pred)
    hd95 = PerformanceMetrics.hd95(y_true, y_pred)

    all_metrics = PerformanceMetrics.evaluate_all_metrics(y_true, y_pred)

