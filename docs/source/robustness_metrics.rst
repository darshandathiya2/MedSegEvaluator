Robustness Metrics
==================

MedSegEvaluator computes:

- **Dice Score Drop (DSD)** – change in Dice due to perturbation.
- **Global Robustness Score (GRS)** – summarises model stability across all perturbations.

GRM Definition:

.. math::

    GRS = \frac{Dice + HD95 + CCC}{3}
