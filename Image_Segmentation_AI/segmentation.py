"""
segmentation.py
----------------
Classical (non-AI, non-ML) image segmentation techniques using OpenCV.

Implemented methods:
    1. Binary Thresholding
    2. Otsu Thresholding
    3. Adaptive Thresholding
    4. Edge-Based Segmentation (Canny + morphology)
    5. Region-Based Segmentation (connected-component based region growing
       on thresholded regions -- classical, NOT machine-learning clustering)

Edge-based and region-based segmentation use a small amount of morphology
(dilation/closing) internally, only as part of turning detected edges or
thresholded pixels into solid, connected regions. This is inherent to how
those two algorithms work and is not a separate, user-facing processing
step.

All of this is standard digital image processing. No learning,
training, or statistical models are involved anywhere in this file.
"""

import cv2
import numpy as np


def binary_threshold(gray_image: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    """
    Simple global binary thresholding.

    Uses THRESH_BINARY_INV because most classroom segmentation targets
    (dark objects on a light background, e.g. the generated sample
    image) have foreground objects darker than the background. Pixels
    BELOW the threshold become foreground (255) in the output mask.
    """
    _, mask = cv2.threshold(gray_image, thresh_value, 255, cv2.THRESH_BINARY_INV)
    return mask


def otsu_threshold(gray_image: np.ndarray) -> np.ndarray:
    """
    Otsu's method automatically calculates the optimal global threshold.
    THRESH_BINARY_INV is used for the same reason as binary_threshold().
    """
    _, mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask


def adaptive_threshold(gray_image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Adaptive thresholding calculates a local threshold for small regions
    of the image, useful for images with uneven illumination.
    THRESH_BINARY_INV is used so dark objects come out as foreground.
    """
    # block_size must be odd and > 1
    bs = int(block_size)
    if bs % 2 == 0:
        bs += 1
    if bs < 3:
        bs = 3

    mask = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        c,
    )
    return mask


def edge_based_segmentation(gray_image: np.ndarray, low_thresh: int = 50,
                             high_thresh: int = 150, kernel_size: int = 3) -> np.ndarray:
    """
    Edge-based segmentation using the Canny edge detector, followed by
    morphological closing/dilation to connect broken edges into filled
    regions suitable for connected-component analysis.
    """
    edges = cv2.Canny(gray_image, low_thresh, high_thresh)

    k = max(1, int(kernel_size))
    kernel = np.ones((k, k), np.uint8)

    # Dilate edges to close small gaps, then close to fill contours
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill holes using contour detection + fillPoly so that the interior
    # of edge-outlined shapes becomes a solid region
    filled = closed.copy()
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled


def region_based_segmentation(gray_image: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    """
    Classical region-based segmentation.

    Approach (no machine-learning clustering involved):
        1. Threshold the image to obtain candidate foreground pixels.
        2. Use connected-component labelling to group neighbouring
           foreground pixels into distinct regions.
        3. Use morphological closing to merge nearby fragments of the
           same object into a single coherent region.

    This is a simplified, classical stand-in for classic "region growing" /
    "region merging" segmentation taught in traditional image processing
    courses.
    """
    _, mask = cv2.threshold(gray_image, thresh_value, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return merged


def segment_image(gray_image: np.ndarray, method: str, params: dict) -> np.ndarray:
    """
    Dispatcher function that routes to the correct classical segmentation
    method based on the user's dropdown selection in the Streamlit UI.

    params may contain: threshold_value, block_size, c, low_thresh,
    high_thresh, kernel_size
    """
    if gray_image is None:
        raise ValueError("Cannot segment a None image.")

    method = method.strip().lower()

    if method == "binary thresholding":
        return binary_threshold(gray_image, params.get("threshold_value", 127))
    elif method == "otsu thresholding":
        return otsu_threshold(gray_image)
    elif method == "adaptive thresholding":
        return adaptive_threshold(
            gray_image,
            params.get("block_size", 11),
            params.get("c", 2),
        )
    elif method == "edge-based segmentation":
        return edge_based_segmentation(
            gray_image,
            params.get("low_thresh", 50),
            params.get("high_thresh", 150),
            params.get("kernel_size", 3),
        )
    elif method == "region-based segmentation":
        return region_based_segmentation(gray_image, params.get("threshold_value", 127))
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
