MedSegEvaluator Documentation
=============================

Welcome to the official documentation of **MedSegEvaluator**, a modular Python library designed for comprehensive evaluation of medical image segmentation models, focusing on accuracy and robustness model. It provides an easy-to-use framework to assess segmentation performance across multiple dimensions, from voxel-level similarity to volumetric and morphological consistency.

MedSegEvaluator provides:

- Slice-level and volume-level Dice computation
- Support for multiple perturbations to assess model robustness
- Global Robustness Score (GRS)
- Visualization utilities
- Easy integration into any deep learning pipeline

------------
Installation
------------

Install the package using pip::

    pip install medsegevaluator


------------
Documentation Contents
------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   usage
   med_seg_metrics
   perturbation
   evaluatation_pipeline
   robustness_metrics
   performance_visualization
   SegmentationPerformancePrecision
   sample
   api_reference

-------------
API Reference
-------------

Full API documentation for MedSegEvaluator.

