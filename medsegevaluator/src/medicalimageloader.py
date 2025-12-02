import os
import numpy as np
import pydicom
import nibabel as nib
import cv2

class MedicalImageLoader:
    """
    Unified medical image loader for DICOM (.dcm), NIfTI (.nii, .nii.gz), and image (.png, .jpg) formats.
    Works for both images and segmentation masks.
    """

    def __init__(self, normalize=True):
        """
        Args:
            normalize (bool): Whether to normalize image intensity to [0, 1].
        """
        self.normalize = normalize

    def load_image(self, path):
        """
        Load a medical image or mask file based on extension.
        Returns a numpy array (2D or 3D).
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".dcm":
            return self._load_dicom(path)
        elif ext in [".nii", ".gz"]:
            return self._load_nifti(path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return self._load_png_jpg(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_dicom(self, path):
        """Load a DICOM image and convert to numpy array."""
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        if self.normalize:
            img = self._normalize(img)
        return img

    def _load_nifti(self, path):
        """Load a NIfTI image and return numpy array."""
        nifti = nib.load(path)
        img = nifti.get_fdata().astype(np.float32)

        if self.normalize:
            img = self._normalize(img)
        return img

    def _load_png_jpg(self, path):
        """Load a PNG/JPG image and return grayscale float32 array."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Handle masks with 3 channels by converting to single channel
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        if self.normalize:
            img = self._normalize(img)
        return img

    def _normalize(self, img):
        """Normalize image to [0, 1] range."""
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    def load_pair(self, image_path, mask_path):
        """
        Load an image and its corresponding mask.
        Returns: (image_array, mask_array)
        """
        image = self.load_image(image_path)
        mask = self.load_image(mask_path)

        # Resize mask if mismatch
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return image, mask

