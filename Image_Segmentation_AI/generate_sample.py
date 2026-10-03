"""
generate_sample.py
-------------------
Utility script to programmatically generate a simple high-contrast
sample image (white background with multiple dark geometric shapes).

This is NOT part of the AI pipeline. It is only used once to create
input_images/sample.jpg so the project can be tested without needing
an external dataset.

Run:
    python generate_sample.py
"""

import cv2
import numpy as np
import os


def generate_sample_image(path="input_images/sample.jpg"):
    # White background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Dark geometric objects of varying size (so rule engine has
    # both "relevant" and "irrelevant" regions to reason about)

    # Large rectangle
    cv2.rectangle(img, (40, 40), (140, 120), (30, 30, 30), -1)

    # Large circle
    cv2.circle(img, (250, 90), 55, (20, 20, 20), -1)

    # Medium triangle
    pts = np.array([[380, 150], [440, 40], [500, 150]], np.int32)
    cv2.fillPoly(img, [pts], (40, 40, 40))

    # Medium rectangle
    cv2.rectangle(img, (60, 220), (180, 300), (25, 25, 25), -1)

    # Small square (should be rejected by default rules - small area)
    cv2.rectangle(img, (250, 250), (270, 270), (35, 35, 35), -1)

    # Small circle (should be rejected by default rules - small area)
    cv2.circle(img, (350, 260), 8, (15, 15, 15), -1)

    # Medium ellipse
    cv2.ellipse(img, (480, 280), (60, 35), 0, 0, 360, (45, 45, 45), -1)

    # Tiny noise dot (should be rejected)
    cv2.circle(img, (520, 350), 3, (10, 10, 10), -1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Sample image saved to: {path}")


if __name__ == "__main__":
    generate_sample_image()
