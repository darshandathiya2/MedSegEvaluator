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

Dice Coefficient
-----------------

.. automethod:: dice_score

   Compute the Dice coefficient (F1 score) for binary segmentation.
   
   .. math::
      \text{Dice} = \frac{2 \cdot |Y_{\text{true}} \cap Y_{\text{pred}}|}{|Y_{\text{true}}| + |Y_{\text{pred}}|}
   
   **Parameters:**
   
   * **y_true** (*numpy.ndarray*) – Ground truth binary mask
   * **y_pred** (*numpy.ndarray*) – Predicted binary mask
   
   **Returns:** Dice coefficient between 0 and 1
   
   **Return type:** float
   
   **Example:**
   
   .. code-block:: python
   
      dice = PerformanceMetrics.dice_score(ground_truth, prediction)
      print(f"Dice Score: {dice:.3f}")
