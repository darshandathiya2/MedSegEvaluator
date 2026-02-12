Model Robustness Evaluation Under Perturbation
==============================================

Typical Workflow
----------------

1. Load input images and corresponding ground-truth masks.
2. Compute baseline performance metrics (Dice, HD95, IoU, etc.).
3. Visualize metric distributions using histograms, boxplots, and confidence intervals.
4. Apply input perturbations and re-evaluate model performance.
5. Compare baseline and perturbed results using robustness visualizations.
6. Analyze performance stability and distribution shifts (e.g., using PSI).
7. Inspect failure cases and uncertain predictions.
