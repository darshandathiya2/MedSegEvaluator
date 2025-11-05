# MedSegEvaluator

MedSegEvaluator is a modular Python library designed for comprehensive evaluation of medical image segmentation models, focusing on accuracy, robustness, and image quality. It provides an easy-to-use framework to assess segmentation performance across multiple dimensions — from voxel-level similarity to volumetric and morphological consistency.

## Key Feature

- **Segmentation Metrics**
  - Dice, IoU, Hausdorff Distance (HD95), Average Surface Distance (ASD)
- **Morphological Analysis**
  - Lesion volume, connected component (island) count, and 2D/3D area evaluation
- **Visualization Tools**
  - Contour overlays, histograms, and boxplots for metric distributions
- **Robustness Assessment**
  - Evaluate how segmentation quality changes under image perturbations or quality degradation

## Applications

- Quantitative comparison of segmentation models (like UNet, nnUNet, DeepLab, etc.)
- Robustness evaluation of models trained under noise, resolution, or intensity variations.
- Exploratory analysis of segmentation consistency across lesion sizes and connectivity patterns.

## Why MedSegEvaluator?

While traditional tools stop at Dice or IoU, **MedSegEvaluator** goes beyond by incorporating:
- Structural metrics (ASD, HD95)  
- Morphological statistics (volume, island count)
- Visulization of Contour overlays, histograms, and boxplots for metric distributions

This makes it ideal for **robustness studies**, **comparative model evaluation**, and **clinical reproducibility** research.




