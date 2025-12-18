Image Perturbations
===================

MedSegEvaluator supports:

- Gaussian Noise  
- Gaussian Blur  
- Salt-and-Pepper Noise  
- Brightness Shift  
- Contrast Shift  
- Rotation (90°, 180°, 270°)  
- Horizontal Flip  
- Vertical Flip  

Example:

.. code-block:: python

    from image_perturbation import apply_blur

    blur_image = apply_blur(image, ksize=5)

.. automodule:: medsegevaluator.image_perturbation
   :members:
   :undoc-members:
   :show-inheritance:
