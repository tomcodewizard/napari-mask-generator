import numpy as np
from skimage import exposure, filters


def preprocess_image(
        image: np.ndarray,
        gaussian_sigma: float = 1.0,
        use_clahe: bool = True
) -> np.ndarray:
    """
    Denoise and enhance contrast for adipocyte images.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    gaussian_sigma : float
        Sigma for Gaussian denoising.
    use_clahe : bool
        Whether to apply CLAHE contrast enhancement.

    Returns
    -------
    np.ndarray
        Preprocessed image normalised to [0, 1].
    """
    # Normalise to float [0, 1]
    image = image.astype(float)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Gaussian denoising
    if gaussian_sigma > 0:
        image = filters.gaussian(image, sigma=gaussian_sigma)

    # CLAHE contrast enhancement
    if use_clahe:
        image = exposure.equalize_adapthist(image, clip_limit=0.03)

    return image