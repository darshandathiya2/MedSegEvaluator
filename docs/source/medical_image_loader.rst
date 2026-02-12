Medical Image Loader
====================

.. module:: medical_image_loader

The ``medical_image_loader`` module provides a unified interface for loading
medical images and segmentation masks in multiple formats. It is designed
for use in model evaluation and robustness analysis pipelines.

Supported file formats include:

- DICOM (``.dcm``)
- NIfTI (``.nii``, ``.nii.gz``)
- PNG / JPG (``.png``, ``.jpg``, ``.jpeg``)


Overview
--------

The ``MedicalImageLoader`` class enables consistent loading, preprocessing,
and normalization of medical images across different modalities. It ensures
that images and corresponding segmentation masks are spatially aligned
before evaluation.


Class Reference
---------------

MedicalImageLoader
^^^^^^^^^^^^^^^^^^

.. class:: MedicalImageLoader(normalize=True)

   Unified loader for medical images and segmentation masks.

   :param bool normalize: If True, normalize image intensities to [0, 1].
   :default normalize: True


Methods
-------

load_image
~~~~~~~~~~

.. method:: MedicalImageLoader.load_image(path)

   Load a medical image or segmentation mask.

   Supported formats include DICOM, NIfTI, and standard image files.

   :param str path: Path to the image or mask file.
   :returns: Loaded image as a NumPy array.
   :rtype: numpy.ndarray
   :raises ValueError: If the file format is unsupported.
   :raises FileNotFoundError: If the file does not exist.


load_pair
~~~~~~~~~

.. method:: MedicalImageLoader.load_pair(image_path, mask_path)

   Load an image and its corresponding segmentation mask.

   If the image and mask dimensions differ, the mask is resized using
   nearest-neighbor interpolation.

   :param str image_path: Path to the image file.
   :param str mask_path: Path to the mask file.
   :returns: Image and mask arrays.
   :rtype: tuple(numpy.ndarray, numpy.ndarray)


Normalization
~~~~~~~~~~~~~

.. method:: MedicalImageLoader._normalize(img)

   Normalize image intensities to the range [0, 1].

   :param numpy.ndarray img: Input image.
   :returns: Normalized image.
   :rtype: numpy.ndarray


DICOM Loading
~~~~~~~~~~~~~

.. method:: MedicalImageLoader._load_dicom(path)

   Load and preprocess a DICOM image.

   Rescale slope and intercept are applied when available.

   :param str path: Path to the DICOM file.
   :returns: DICOM image array.
   :rtype: numpy.ndarray


NIfTI Loading
~~~~~~~~~~~~~

.. method:: MedicalImageLoader._load_nifti(path)

   Load a NIfTI image.

   :param str path: Path to the NIfTI file.
   :returns: NIfTI image array.
   :rtype: numpy.ndarray


PNG / JPG Loading
~~~~~~~~~~~~~~~~~

.. method:: MedicalImageLoader._load_png_jpg(path)

   Load a PNG or JPG image and convert to grayscale if necessary.

   :param str path: Path to the image file.
   :returns: Image array.
   :rtype: numpy.ndarray


Usage Example
-------------

The following example demonstrates how to use the loader in an evaluation
pipeline.

.. code-block:: python

   from medical_image_loader import MedicalImageLoader

   loader = MedicalImageLoader(normalize=True)

   image, mask = loader.load_pair(
       "data/sample_image.nii.gz",
       "data/sample_mask.nii.gz"
   )

   # Use image and mask for inference and metric computation


Integration in Robustness Analysis
----------------------------------

This module is intended to be used as the first step in robustness and
perturbation analysis pipelines:

1. Load original images and masks.
2. Apply image perturbations.
3. Generate model predictions.
4. Compute performance metrics (e.g., Dice score).
5. Analyze distribution shifts using stability measures (e.g., PSI).

The standardized loading process ensures fair and reproducible
model evaluation across different perturbation settings.


Dependencies
------------

This module requires the following Python packages:

- ``numpy``
- ``pydicom``
- ``nibabel``
- ``opencv-python``


License
-------

This module is intended for research and academic use.
