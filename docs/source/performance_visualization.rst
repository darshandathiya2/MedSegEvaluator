Performance Visualization
==========================

This page documents MedSegEvaluator's visualization tools for exploring metric distributions nd agreement: **Histogram**, **Box plot**, **Bland-Altman plot** and **Model's performance under perturbations**. These visualization helps you to inspect model behaviour across a dataset, compare models/perturbations, and identify bias or outlier.

.. contents::
    :local:
    :depth: 2

MedSegEvaluator provides three primary visualization functions:

- **Histogram:** visualize distribution of a metric (Dice, IoU, etc.).  
- **Box Plot:** compare distributions across groups (models, perturbations).  
- **Bland–Altman Plot:** — analyze agreement and bias between two sets of measurements (e.g., GT vs prediction, or model A vs                                          model B).



