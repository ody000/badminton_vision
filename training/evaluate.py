"""Shared evaluation metrics for all training scripts.

Provides:
  - compute_distance_error(): mean pixel distance between heatmap peaks and GT centers
  - compute_iou(): standard IoU between two bounding boxes
  - compute_map_at_50(): simple mAP@0.5
  - evaluate_tracknet_epoch(): run TrackNet model over a dataloader
  - evaluate_yolo_epoch(): wrap Ultralytics results object
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Primitive metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_distance_error(
    pred_centers: list,
    gt_centers: list,
) -> float:
    """Mean Euclidean pixel distance between predicted heatmap peaks and GT centers.

    Args:
        pred_centers: List of (x, y) tuples or None for each sample.
        gt_centers: List of (x, y) tuples or None for each sample.

    Returns:
        Mean distance in pixels (ignores samples where either is None).
    """
    errors = []
    for pred, gt in zip(pred_centers, gt_centers):
        if pred is None or gt is None:
            continue
        dist = math.hypot(float(pred[0]) - float(gt[0]), float(pred[1]) - float(gt[1]))
        errors.append(dist)
    if not errors:
        return 0.0
    return float(sum(errors) / len(errors))


def compute_iou(box_a: list, box_b: list) -> float:
    """Standard IoU between two [x1, y1, x2, y2] boxes.

    Returns value in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_map_at_50(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Simple mAP@0.5 computation.

    Args:
        predictions: List of {"box": [x1,y1,x2,y2], "score": float, "class": int}.
        ground_truths: List of {"box": [x1,y1,x2,y2], "class": int}.
        iou_threshold: IoU threshold to count a prediction as TP.

    Returns:
        mAP@0.5 as float in [0.0, 1.0].
    """
    if not predictions or not ground_truths:
        return 0.0

    # Group by class
    classes = set(p["class"] for p in predictions) | set(g["class"] for g in ground_truths)
    aps = []

    for cls in classes:
        preds_cls = sorted(
            [p for p in predictions if p["class"] == cls],
            key=lambda x: -x["score"],
        )
        gts_cls = [g for g in ground_truths if g["class"] == cls]

        if not gts_cls:
            continue

        matched_gts = set()
        tp = []
        fp = []

        for pred in preds_cls:
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt in enumerate(gts_cls):
                if gi in matched_gts:
                    continue
                iou = compute_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched_gts.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall = tp_cum / (len(gts_cls) + 1e-9)

        # AP via trapezoidal rule
        ap = float(np.trapz(precision, recall)) if len(recall) > 1 else float(precision[-1] if len(precision) else 0.0)
        aps.append(ap)

    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))


# ─────────────────────────────────────────────────────────────────────────────
# Epoch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tracknet_epoch(model, dataloader, device, cfg) -> dict:
    """Run TrackNet model over a validation dataloader and return metrics.

    Returns:
        {"distance_error_px": float, "map_50": float}
    """
    import torch
    import cv2

    model.eval()
    box_size = int(getattr(cfg, "tracknet_box_size", 16))

    pred_centers = []
    gt_centers = []
    preds_for_map = []
    gts_for_map = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)  # (B, 9, H, W)
            targets = batch["target"].to(device)  # (B, 1, H, W) Gaussian heatmap

            outputs = model(inputs)  # (B, C, H, W)
            # Use channel 0 for heatmap
            out_hm = outputs[:, 0, :, :].cpu().numpy()
            tgt_hm = targets[:, 0, :, :].cpu().numpy()

            for i in range(out_hm.shape[0]):
                ph = out_hm[i]
                gh = tgt_hm[i]

                _, _, _, pred_loc = cv2.minMaxLoc(ph.astype(np.float32))
                _, _, _, gt_loc = cv2.minMaxLoc(gh.astype(np.float32))

                pred_centers.append(pred_loc)
                gt_centers.append(gt_loc)

                # Build prediction / GT boxes for mAP
                px, py = pred_loc
                gx, gy = gt_loc
                half = box_size // 2
                pred_box = [px - half, py - half, px + half, py + half]
                gt_box = [gx - half, gy - half, gx + half, gy + half]
                preds_for_map.append({"box": pred_box, "score": float(ph.max()), "class": 0})
                gts_for_map.append({"box": gt_box, "class": 0})

    dist_err = compute_distance_error(pred_centers, gt_centers)
    map50 = compute_map_at_50(preds_for_map, gts_for_map)

    return {"distance_error_px": dist_err, "map_50": map50}


def evaluate_yolo_epoch(results) -> dict:
    """Wrap Ultralytics results object and return standard metrics dict.

    Args:
        results: Ultralytics Results object from model.val().

    Returns:
        {"map_50": float, "map_50_95": float, "precision": float, "recall": float}
    """
    try:
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        return {
            "map_50": float(metrics.get("metrics/mAP50(B)", metrics.get("mAP50", 0.0))),
            "map_50_95": float(metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95", 0.0))),
            "precision": float(metrics.get("metrics/precision(B)", metrics.get("precision", 0.0))),
            "recall": float(metrics.get("metrics/recall(B)", metrics.get("recall", 0.0))),
        }
    except Exception as e:
        print(f"[EVAL] evaluate_yolo_epoch error: {e}")
        return {"map_50": 0.0, "map_50_95": 0.0, "precision": 0.0, "recall": 0.0}
