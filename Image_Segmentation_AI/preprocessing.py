"""
preprocessing.py
-----------------
Classical (non-AI) image preprocessing utilities using OpenCV.

Responsibilities:
    1. Reading an image (already decoded array is expected here since
       Streamlit gives us an uploaded file, decoding happens in app.py)
    2. Converting BGR -> RGB (for correct display in Streamlit)
    3. Converting to Grayscale
    4. Applying Gaussian Blur to reduce noise before segmentation

None of this is "AI" -- it is standard classical image processing and
is clearly labelled as such in the README and the UI.
"""

import cv2
import numpy as np


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR (OpenCV default) image to RGB (for display)."""
    if image_bgr is None:
        raise ValueError("Input image is None. Cannot convert to RGB.")
    if len(image_bgr.shape) == 2:
        # Already single channel, just return as-is
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if image_bgr is None:
        raise ValueError("Input image is None. Cannot convert to grayscale.")
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to a grayscale image to reduce noise before
    thresholding / edge detection.

    kernel_size must be a positive odd integer.
    """
    if gray_image is None:
        raise ValueError("Input image is None. Cannot apply Gaussian blur.")

    # Gaussian kernel size must be odd and >= 1
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    if k < 1:
        k = 1

    return cv2.GaussianBlur(gray_image, (k, k), 0)


def preprocess_pipeline(image_bgr: np.ndarray, enable_preprocessing: bool = True,
                         blur_kernel: int = 5):
    """
    Full classical preprocessing pipeline.

    Steps:
        1. RGB conversion (for display only)
        2. Grayscale conversion
        3. Gaussian blur (optional, controlled by enable_preprocessing)

    Returns:
        dict with keys: 'rgb', 'gray', 'preprocessed'
    """
    if image_bgr is None:
        raise ValueError("Input image is None. Please upload a valid image.")

    if image_bgr.size == 0:
        raise ValueError("Input image is empty (0 bytes / 0 pixels).")

    rgb = to_rgb(image_bgr)
    gray = to_grayscale(image_bgr)

    if enable_preprocessing:
        preprocessed = apply_gaussian_blur(gray, blur_kernel)
    else:
        preprocessed = gray.copy()

    return {
        "rgb": rgb,
        "gray": gray,
        "preprocessed": preprocessed,
    }
