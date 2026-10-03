"""
evaluation.py
-------------
Optional quantitative segmentation evaluation against a ground-truth mask.

If the user provides a ground-truth binary mask (same size as the
predicted mask), this module computes standard, well-known image
segmentation metrics:

    - IoU (Intersection over Union)
    - Dice Coefficient
    - Pixel Accuracy

These are simple set-based arithmetic formulas -- not learned metrics --
so they remain fully consistent with the "no ML/DL" requirement of this
project.

If no ground-truth mask is available, no accuracy numbers are invented.
The UI must instead display:

    "Ground-truth mask not provided. Quantitative segmentation accuracy
    cannot be calculated."
"""

import numpy as np


def _validate_masks(pred_mask: np.ndarray, gt_mask: np.ndarray):
    if pred_mask is None or gt_mask is None:
        raise ValueError("Both predicted mask and ground-truth mask are required.")
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted mask {pred_mask.shape} vs "
            f"ground-truth mask {gt_mask.shape}. They must match exactly."
        )


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is strictly boolean (foreground = pixel > 0)."""
    return (mask > 0).astype(np.uint8)


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection over Union:
        IoU = |Pred ∩ GroundTruth| / |Pred ∪ GroundTruth|
    """
    _validate_masks(pred_mask, gt_mask)
    pred = _binarize(pred_mask)
    gt = _binarize(gt_mask)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return round(float(intersection) / float(union), 4)


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice Coefficient (F1 score for segmentation):
        Dice = 2 * |Pred ∩ GroundTruth| / (|Pred| + |GroundTruth|)
    """
    _validate_masks(pred_mask, gt_mask)
    pred = _binarize(pred_mask)
    gt = _binarize(gt_mask)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return round((2.0 * float(intersection)) / float(total), 4)


def calculate_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Pixel Accuracy:
        PA = (correctly classified pixels) / (total pixels)
    """
    _validate_masks(pred_mask, gt_mask)
    pred = _binarize(pred_mask)
    gt = _binarize(gt_mask)

    correct = (pred == gt).sum()
    total_pixels = pred.size

    return round(float(correct) / float(total_pixels), 4)


def evaluate_segmentation(pred_mask: np.ndarray, gt_mask: np.ndarray = None) -> dict:
    """
    Main evaluation entry point used by app.py.

    If gt_mask is None, returns a dict signalling that no ground-truth
    mask was provided -- no fake/invented accuracy values are ever
    returned.
    """
    if gt_mask is None:
        return {
            "available": False,
            "message": (
                "Ground-truth mask not provided. Quantitative segmentation "
                "accuracy cannot be calculated."
            ),
        }

    iou = calculate_iou(pred_mask, gt_mask)
    dice = calculate_dice(pred_mask, gt_mask)
    pixel_acc = calculate_pixel_accuracy(pred_mask, gt_mask)

    return {
        "available": True,
        "iou": iou,
        "dice_coefficient": dice,
        "pixel_accuracy": pixel_acc,
    }
