Image Perturbations
===================

The ``ImagePerturbation`` module provides a collection of deterministic and stochastic image transformations designed to simulate common acquisition and preprocessing variations encountered in medical imaging pipelines. All perturbations operate on normalized images in the range [0,1].

The module implements the following image perturbation types:

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
