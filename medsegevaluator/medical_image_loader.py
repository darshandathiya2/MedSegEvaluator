import os
import numpy as np
import pydicom
import nibabel as nib
import cv2


class MedicalImageLoader:
    """
    Unified medical image loader for DICOM (.dcm), NIfTI (.nii, .nii.gz),
    and standard image (.png, .jpg, .jpeg) formats.

    Designed for loading both medical images and segmentation masks
    in evaluation and robustness analysis pipelines.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the loader.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize image intensities to [0, 1].
            Default is True.
        """
        self.normalize = normalize

    def load_image(self, path: str) -> np.ndarray:
        """
        Load a medical image or mask file based on extension.

        Supported formats:
        - DICOM (.dcm)
        - NIfTI (.nii, .nii.gz)
        - PNG/JPG (.png, .jpg, .jpeg)

        Parameters
        ----------
        path : str
            Path to the image or mask file.

        Returns
        -------
        np.ndarray
            Loaded image as a NumPy array (2D or 3D).

        Raises
        ------
        ValueError
            If the file format is unsupported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".dcm":
            return self._load_dicom(path)

        elif ext in [".nii", ".gz"]:
            return self._load_nifti(path)

        elif ext in [".png", ".jpg", ".jpeg"]:
            return self._load_png_jpg(path)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_pair(self, image_path: str, mask_path: str):
        """
        Load an image and its corresponding segmentation mask.

        If image and mask shapes differ, the mask is resized
        using nearest-neighbor interpolation.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        mask_path : str
            Path to the segmentation mask.

        Returns
        -------
        tuple
            (image_array, mask_array)
        """
        image = self.load_image(image_path)
        mask = self.load_image(mask_path)

        if image.shape != mask.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        return image, mask

    def _load_dicom(self, path: str) -> np.ndarray:
        """
        Load a DICOM image and apply rescale slope/intercept.

        Parameters
        ----------
        path : str
            Path to the DICOM file.

        Returns
        -------
        np.ndarray
            DICOM image as float32 NumPy array.
        """
        dcm = pydicom.dcmread(path)

        img = dcm.pixel_array.astype(np.float32)

        # Apply rescale slope and intercept
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        if self.normalize:
            img = self._normalize(img)

        return img

    def _load_nifti(self, path: str) -> np.ndarray:
        """
        Load a NIfTI image.

        Parameters
        ----------
        path : str
            Path to the NIfTI file.

        Returns
        -------
        np.ndarray
            NIfTI image as float32 NumPy array.
        """
        nifti = nib.load(path)

        img = nifti.get_fdata().astype(np.float32)

        if self.normalize:
            img = self._normalize(img)

        return img

    def _load_png_jpg(self, path: str) -> np.ndarray:
        """
        Load a PNG/JPG image and convert to grayscale if needed.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            Grayscale image as float32 NumPy array.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert RGB/BGR to grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)

        if self.normalize:
            img = self._normalize(img)

        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Parameters
        ----------
        img : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Normalized image.
        """
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)

        return img
