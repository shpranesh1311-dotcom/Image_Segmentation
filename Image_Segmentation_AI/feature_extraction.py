"""
feature_extraction.py
----------------------
Classical connected-component analysis and geometric feature extraction.

Uses cv2.connectedComponentsWithStats() to detect individual regions in a
binary mask (produced by segmentation.py) and computes simple, explainable
geometric features for each region:

    - Region ID
    - Area
    - Width
    - Height
    - Bounding box (x, y, w, h)
    - Centroid (cx, cy)
    - Aspect ratio

These features are NOT learned by any model -- they are computed directly
from pixel statistics returned by OpenCV's connected-component algorithm.
The features are then handed to rule_engine.py, which is where the
traditional AI reasoning happens.
"""

import cv2
import numpy as np
import pandas as pd


def extract_regions(binary_mask: np.ndarray) -> pd.DataFrame:
    """
    Run connected-component analysis on a binary mask and extract
    geometric features for every foreground region.

    Background (label 0) is always ignored.

    Returns a pandas DataFrame with columns:
        region_id, area, width, height, x, y, centroid_x, centroid_y, aspect_ratio
    """
    if binary_mask is None:
        raise ValueError("Cannot extract regions from a None mask.")

    # connectedComponentsWithStats requires an 8-bit single-channel image
    mask = binary_mask
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    records = []

    # label 0 is always the background -> skip it
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]

        aspect_ratio = round(w / h, 2) if h > 0 else 0.0

        records.append({
            "region_id": label,
            "area": area,
            "width": w,
            "height": h,
            "x": x,
            "y": y,
            "centroid_x": round(float(cx), 1),
            "centroid_y": round(float(cy), 1),
            "aspect_ratio": aspect_ratio,
        })

    df = pd.DataFrame(records)
    return df, labels, num_labels
