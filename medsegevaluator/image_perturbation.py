import cv2
import numpy as np


class ImagePerturbation:

    @staticmethod
    def _normalize(image):
        r"""
            Normalize image intensities to the range ``[0, 1]``.
        
            Let the input image be denoted as :math:`I \in \mathbb{R}^{H \times W \times C}`,
            where pixel intensities are typically represented in the range ``[0, 255]``.
            The normalization is defined as:
        
            .. math::
        
                I_{\text{norm}} =
                \begin{cases}
                    \dfrac{I}{255}, & \text{if } \max(I) > 1 \\
                    I, & \text{otherwise}
                \end{cases}
        
            This conditional scaling ensures numerical stability and guarantees that
            all subsequent perturbation operations are applied to images with values
            bounded within ``[0, 1]``.
        
            Parameters
            ----------
            image : numpy.ndarray
                Input image of shape ``(H, W, C)`` or ``(H, W)``.
        
            Returns
            -------
            numpy.ndarray
                Normalized image with floating-point values in the range ``[0, 1]``.
        """
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return image

    
    @staticmethod
    def add_noise(image, noise_type="gaussian", mean=0, var=0.001):
        r"""
        Add noise to a normalized image.
    
        Let the normalized input image be denoted as
        :math:`I \in [0,1]^{H \times W \times C}`.
    
        **Gaussian Noise**
    
        When ``noise_type="gaussian"``, additive Gaussian noise is applied:
    
        .. math::
    
            I_{\text{noisy}} = \text{clip}(I + N, 0, 1)
    
        where
    
        .. math::
    
            N \sim \mathcal{N}(\mu, \sigma^2), \quad \sigma = \sqrt{\text{var}}
    
        and :math:`\mu` is the mean of the noise distribution.
    
        **Salt-and-Pepper Noise**
    
        When ``noise_type="salt_pepper"``, a fraction of pixels is randomly
        replaced with extreme intensity values:
    
        .. math::
    
            I_{\text{noisy}}(x) =
            \begin{cases}
                1, & \text{with probability } p_{\text{salt}} \\
                0, & \text{with probability } p_{\text{pepper}} \\
                I(x), & \text{otherwise}
            \end{cases}
    
        where the total fraction of corrupted pixels is controlled by
        ``amount``, and salt-to-pepper ratio is defined by ``s\_vs\_p``.
    
        All output values are clipped to ensure they remain within
        the valid intensity range ``[0, 1]``.
    
        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape ``(H, W, C)`` or ``(H, W)``.
        noise_type : str
            Type of noise to apply: ``"gaussian"`` or ``"salt_pepper"``.
        mean : float
            Mean of the Gaussian noise distribution.
        var : float
            Variance of the Gaussian noise distribution.
    
        Returns
        -------
        numpy.ndarray
            Noise-perturbed image with values in the range ``[0, 1]``.
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
        r"""
        Adjust image brightness by intensity scaling.
    
        Let the normalized input image be denoted as
        :math:`I \in [0,1]^{H \times W \times C}`.
        Brightness adjustment is performed via linear scaling:
    
        .. math::
    
            I_{\text{bright}} = \text{clip}(\alpha \cdot I, 0, 1)
    
        where :math:`\alpha` is the brightness scaling factor.
        Values of :math:`\alpha > 1` increase brightness, while
        :math:`0 < \alpha < 1` decrease brightness.
    
        The clipping operation ensures that all pixel intensities
        remain within the valid range ``[0, 1]``.
    
        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape ``(H, W, C)`` or ``(H, W)``.
        factor : float, optional
            Brightness scaling factor :math:`\alpha`. Default is ``1.2``.
    
        Returns
        -------
        numpy.ndarray
            Brightness-adjusted image with values in the range ``[0, 1]``.
        """
        image = ImagePerturbation._normalize(image)
        return np.clip(image * factor, 0, 1)

    @staticmethod
    def apply_gaussian_blur(image, ksize=(5, 5), sigma=0):
        r"""
        Applies Gaussian blur to a normalized float32 image [0,1].
        """
        image = ImagePerturbation._normalize(image)
        blurred = cv2.GaussianBlur(image, ksize, sigma)
        return np.clip(blurred, 0, 1)

    @staticmethod
    def apply_rotation(image, angle=90):
        r"""
        Applies rotation (must be 90, 180, or 270) to a normalized float32 image [0,1].
        """
        if angle not in [90, 180, 270]:
            raise ValueError("Angle must be 90, 180, or 270 degrees.")
        return np.rot90(image, k=angle // 90)

    @staticmethod
    def perturb_image(image, perturb_type):
        r"""Apply a single perturbation to an image."""
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

