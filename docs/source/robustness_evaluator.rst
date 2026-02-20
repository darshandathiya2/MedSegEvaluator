Robustness Evaluator
===================

.. module:: medsegevaluator.robustness_evaluator

The ``robustness_evaluator`` module provides tools for evaluating the
robustness of medical image segmentation models under systematic
image perturbations.

It quantifies performance degradation using Dice score and
perturbation-based analysis.


Overview
--------

This module implements a standardized robustness evaluation pipeline:

1. Load original images and masks.
2. Apply predefined perturbations.
3. Generate predictions.
4. Compute Dice scores.
5. Measure performance degradation.

The resulting metrics allow quantitative assessment of model stability.


Class Reference
---------------

RobustnessEvaluator
^^^^^^^^^^^^^^^^^^^

.. class:: RobustnessEvaluator(model, loader=None, threshold=0.5)


   Main class for robustness evaluation.

   :param model: Trained segmentation model.
   :type model: tensorflow.keras.Model

   :param loader: Medical image loader instance.
   :type loader: MedicalImageLoader, optional

   :param float threshold: Binarization threshold for predictions.
   :default threshold: 0.5


Perturbation Methods
--------------------

The evaluator applies the following perturbations:

- Gaussian noise
- Gaussian blur
- Brightness adjustment
- Salt-and-pepper noise
- Rotation (90°, 180°, 270°)
- Horizontal flip
- Vertical flip

Geometric perturbations are applied to both images and masks, while
photometric perturbations are applied only to images.


Evaluation Method
-----------------

evaluate
~~~~~~~~

.. method:: RobustnessEvaluator.evaluate(image_paths, mask_paths, output_csv=None)

   Run robustness analysis on a dataset.

   :param list image_paths: List of image file paths.
   :param list mask_paths: List of mask file paths.
   :param str output_csv: Optional output CSV path.
   :returns: Evaluation results.
   :rtype: pandas.DataFrame


Output Format
-------------

The resulting DataFrame contains:

- ``Image_ID`` : Unique image identifier
- ``Dice_original`` : Dice score on original image
- ``Dice_<perturbation>`` : Dice score per perturbation
- ``Mean_Perturb_Dice`` : Mean Dice under perturbations
- ``Delta_Dice`` : Performance drop


Example Usage
-------------

.. code-block:: python

   from tensorflow.keras.models import load_model
   from robustness_evaluator import RobustnessEvaluator
   from medicalimageloader import MedicalImageLoader

   model = load_model("model.keras")

   loader = MedicalImageLoader(normalize=True)

   evaluator = RobustnessEvaluator(
       model=model,
       loader=loader
   )

   df = evaluator.evaluate(
       image_paths=image_paths,
       mask_paths=mask_paths,
       output_csv="robustness.csv"
   )

   print(df.head())


Integration in Evaluation Pipeline
----------------------------------

This module is designed to be integrated into:

- Model benchmarking frameworks
- Perturbation robustness studies
- Clinical reliability analysis
- Distribution shift investigations

It complements traditional test-set evaluation by providing
fine-grained stability measurements.


Dependencies
------------

This module requires:

- ``numpy``
- ``pandas``
- ``opencv-python``
- ``tensorflow``
- ``performance_metrics``
- ``medicalimageloader``


License
-------

This module is intended for academic and research use.
