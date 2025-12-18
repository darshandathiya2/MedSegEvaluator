import cv2
import numpy as np


class ImagePerturbation:
    def __init__(self):
        pass

    @staticmethod
    def _normalize(image):
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return image

    @staticmethod
    def add_noise(image, noise_type="gaussian", mean=0, var=0.001):
        """
        Adds Gaussian or Salt-and-Pepper noise to a normalized float32 image [0,1].
        """
        image = ImagePerturbation._normalize(image)

        if noise_type == "gaussian":
            noise = np.random.normal(mean, var ** 0.5, image.shape).astype(np.float32)
            noisy = np.clip(image + noise, 0, 1)

        elif noise_type == "salt_pepper":
            s_vs_p = 0.5
            amount = 0.02
            noisy = np.copy(image)

            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy[tuple(coords)] = 1.0

            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy[tuple(coords)] = 0.0
        else:
            noisy = image

        return noisy

    @staticmethod
    def apply_brightness(image, factor=1.2):
        """
        Adjust brightness of a normalized float32 image [0,1].
        """
        image = ImagePerturbation._normalize(image)
        return np.clip(image * factor, 0, 1)

    @staticmethod
    def apply_gaussian_blur(image, ksize=(5, 5), sigma=0):
        """
        Applies Gaussian blur to a normalized float32 image [0,1].
        """
        image = ImagePerturbation._normalize(image)
        blurred = cv2.GaussianBlur(image, ksize, sigma)
        return np.clip(blurred, 0, 1)

    @staticmethod
    def apply_rotation(image, angle=90):
        if angle not in [90, 180, 270]:
            raise ValueError("Angle must be 90, 180, or 270 degrees.")
        return np.rot90(image, k=angle // 90)

    @staticmethod
    def perturb_image(image, perturb_type):
        """Apply a single perturbation to an image."""
        if perturb_type == "gaussian_noise":
            return ImagePerturbation.add_noise(image, "gaussian")
        elif perturb_type == "salt_pepper":
            return ImagePerturbation.add_noise(image, "salt_pepper")
        elif perturb_type == "gaussian_blur":
            return ImagePerturbation.apply_gaussian_blur(image)
        elif perturb_type == "brightness":
            return ImagePerturbation.apply_brightness(image)
        elif perturb_type == "rotation_90":
            return ImagePerturbation.apply_rotation(image, 90)
        elif perturb_type == "rotation_180":
            return ImagePerturbation.apply_rotation(image, 180)
        elif perturb_type == "rotation_270":
            return ImagePerturbation.apply_rotation(image, 270)
        else:
            raise ValueError(f"Unknown perturbation: {perturb_type}")

