Evaluation Pipeline
===================

Typical workflow:

1. Load image + ground truth  
2. Generate perturbed copies  
3. Predict using segmentation model  
4. Compute Dice, HD95, and CCC for each perturbation  
5. Compute global robustness metrics  
6. Visualize changes  

