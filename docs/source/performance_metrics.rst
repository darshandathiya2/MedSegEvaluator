.. _performance-metrics-label:

=================
Performance Metrics
=================

This section contains the definitions of the various metrics used for segmentation evaluation.
They are subdivided into the following categories:

* :py:class:`Region-based, Surface-based, Volume-based, and Robustness Metrics <medsegevaluator.PerformanceMetrics.PerformanceMetrics>` (all in one class)
* :py:class:`Slice-level Metrics <medsegevaluator.PerformanceMetrics.PerformanceMetrics>` (3D)
* :py:class:`Utility Metrics <medsegevaluator.PerformanceMetrics.PerformanceMetrics>`  

All metrics can be applied to predicted and ground truth masks to evaluate segmentation performance.
Surface-based metrics take voxel spacing into account if provided. Slice-level metrics compute statistics
per slice along a specified axis. Utility metrics include the Concordance Correlation Coefficient (CCC).

.. _performance-metrics-class-label:

PerformanceMetrics Class
-----------------------

.. automodule:: medsegevaluator.PerformanceMetrics
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
