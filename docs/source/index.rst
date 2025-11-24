MedSegEvaluator Documentation
=============================

Welcome to the official documentation of **MedSegEvaluator**, a modular Python library designed for comprehensive evaluation of medical image segmentation models, focusing on accuracy and robustness model. It provides an easy-to-use framework to assess segmentation performance across multiple dimensions, from voxel-level similarity to volumetric and morphological consistency.

MedSegEvaluator provides:
\begin{enamurate}
 \item Slice-level and volume-level Dice computation
 \item Support for multiple perturbations to assess model robustness
 \item Global Robustness Score (GRS)
 \item Visualization utilities
 \item Easy integration into any deep learning pipeline
\end{enamurate}
------------
Installation
------------

Install the package using pip::

    pip install medsegevaluator

------------
Quick Start
------------

Here is the simplest example to compute 3D Dice and slice-level Dice::

    from medsegevaluator import dice3d, slice_level_dice

    dice = dice3d(gt, pred)
    slices, stats = slice_level_dice(gt, pred)

    print("3D Dice:", dice)
    print("Slice Stats:", stats)

------------
Documentation Contents
------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   evaluation
   robustness
   visualization
   api_reference

-------------
API Reference
-------------

Full API documentation for MedSegEvaluator.

