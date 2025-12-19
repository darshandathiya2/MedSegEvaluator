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
        Apply Gaussian smoothing to a normalized image.
    
        Let the normalized input image be denoted as
        :math:`I \in [0,1]^{H \times W \times C}`.
        Gaussian blurring is performed via convolution with a Gaussian kernel:
    
        .. math::
    
            I_{\text{blur}} = I * G_{\sigma}
    
        where :math:`*` denotes the convolution operator and
        :math:`G_{\sigma}` is a 2D Gaussian kernel defined as:
    
        .. math::
    
            G_{\sigma}(x, y) = \frac{1}{2\pi\sigma^2}
            \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
    
        The kernel size is specified by ``ksize``, and the standard deviation
        :math:`\sigma` is either provided explicitly or inferred automatically
        when ``sigma = 0``.
    
        After filtering, pixel intensities are clipped to ensure all values
        remain within the valid range ``[0, 1]``.
    
        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape ``(H, W, C)`` or ``(H, W)``.
        ksize : tuple of int, optional
            Size of the Gaussian kernel. Default is ``(5, 5)``.
        sigma : float, optional
            Standard deviation of the Gaussian kernel. Default is ``0``.
    
        Returns
        -------
        numpy.ndarray
            Gaussian-blurred image with values in the range ``[0, 1]``.
        """
        image = ImagePerturbation._normalize(image)
        blurred = cv2.GaussianBlur(image, ksize, sigma)
        return np.clip(blurred, 0, 1)


    @staticmethod
    def apply_rotation(image, angle=90):
        r"""
        Apply discrete rotation to a normalized image.
    
        Let the normalized input image be denoted as
        :math:`I \in [0,1]^{H \times W \times C}`.
        The rotation operation is defined as a spatial transformation:
    
        .. math::
    
            I_{\text{rot}}(x, y) = I(R_{\theta}(x, y))
    
        where :math:`R_{\theta}` is a rotation operator corresponding to
        an angle :math:`\theta \in \{90^\circ, 180^\circ, 270^\circ\}`.
    
        In practice, the rotation is implemented as a sequence of
        90-degree counter-clockwise rotations:
    
        .. math::
    
            k = \frac{\theta}{90}
    
        where :math:`k` denotes the number of 90-degree rotations applied.
    
        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape ``(H, W, C)`` or ``(H, W)``.
        angle : int, optional
            Rotation angle in degrees. Must be one of ``90``, ``180``, or ``270``.
    
        Returns
        -------
        numpy.ndarray
            Rotated image with values in the range ``[0, 1]``.
    
        Raises
        ------
        ValueError
            If ``angle`` is not one of ``90``, ``180``, or ``270``.
        """
        if angle not in [90, 180, 270]:
            raise ValueError("Angle must be 90, 180, or 270 degrees.")
        return np.rot90(image, k=angle // 90)


    @staticmethod
    def perturb_image(image, perturb_type):
        r"""
        Apply a specified image perturbation.
    
        This function acts as a unified interface for applying a single
        perturbation to an input image. Based on the provided
        ``perturb_type``, it internally dispatches the image to the
        corresponding perturbation function.
    
        Supported perturbations include intensity-based transformations
        (Gaussian noise, salt-and-pepper noise, brightness adjustment),
        spatial smoothing (Gaussian blur), and discrete spatial
        transformations (rotations by 90°, 180°, or 270°).
    
        The input image is automatically normalized before the selected
        perturbation is applied, ensuring consistent behavior across all
        transformation types.
    
        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape ``(H, W, C)`` or ``(H, W)``.
        perturb_type : str
            Identifier specifying the perturbation to apply. Supported values are:
            ``"gaussian_noise"``, ``"salt_pepper"``, ``"gaussian_blur"``,
            ``"brightness"``, ``"rotation_90"``, ``"rotation_180"``,
            and ``"rotation_270"``.
    
        Returns
        -------
        numpy.ndarray
            Perturbed image with values constrained to the range ``[0, 1]``.
    
        Raises
        ------
        ValueError
            If an unsupported ``perturb_type`` is provided.
        """
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
