from __future__ import annotations

import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.logical_and(pred_b, gt_b).sum(dtype=np.int64)
    union = np.logical_or(pred_b, gt_b).sum(dtype=np.int64)
    pred_sum = pred_b.sum(dtype=np.int64)
    gt_sum = gt_b.sum(dtype=np.int64)

    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pred_voxels": float(pred_sum),
        "gt_voxels": float(gt_sum),
        "intersection": float(intersection),
    }
