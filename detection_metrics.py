"""NumPy utilities and evaluation metrics for object detection.

Bounding-box convention: xyxy = [x_min, y_min, x_max, y_max].
Coordinates are continuous; box area is max(0, x2 - x1) * max(0, y2 - y1).

This module is intentionally independent of VOC2007 dataset loading and inference code.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _as_boxes(boxes: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Convert boxes to float64 array with shape (N, 4), including empty inputs."""
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two xyxy boxes."""
    a = _as_boxes(np.asarray(box_a, dtype=np.float64).reshape(1, -1), "box_a")
    b = _as_boxes(np.asarray(box_b, dtype=np.float64).reshape(1, -1), "box_b")
    if a.shape != (1, 4) or b.shape != (1, 4):
        raise ValueError("box_a and box_b must each contain exactly 4 coordinates")

    left_top = np.maximum(a[0, :2], b[0, :2])
    right_bottom = np.minimum(a[0, 2:], b[0, 2:])
    intersection_wh = np.maximum(0.0, right_bottom - left_top)
    intersection = float(intersection_wh[0] * intersection_wh[1])

    area_a_wh = np.maximum(0.0, a[0, 2:] - a[0, :2])
    area_b_wh = np.maximum(0.0, b[0, 2:] - b[0, :2])
    area_a = float(area_a_wh[0] * area_a_wh[1])
    area_b = float(area_b_wh[0] * area_b_wh[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def pairwise_iou(
    boxes_a: np.ndarray | Sequence[Sequence[float]],
    boxes_b: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return an (N, M) matrix of pairwise IoUs, vectorized over box pairs."""
    a = _as_boxes(boxes_a, "boxes_a")
    b = _as_boxes(boxes_b, "boxes_b")
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[..., 0] * wh[..., 1]

    wh_a = np.maximum(0.0, a[:, 2:] - a[:, :2])
    wh_b = np.maximum(0.0, b[:, 2:] - b[:, :2])
    area_a = wh_a[:, 0] * wh_a[:, 1]
    area_b = wh_b[:, 0] * wh_b[:, 1]
    union = area_a[:, None] + area_b[None, :] - intersection

    result = np.zeros_like(intersection, dtype=np.float64)
    np.divide(intersection, union, out=result, where=union > 0.0)
    return result


def nms(
    boxes: np.ndarray | Sequence[Sequence[float]],
    scores: Sequence[float] | np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Greedy single-class NMS; return kept indices in descending-score order.

    A candidate is suppressed only when IoU is strictly greater than the threshold.
    Equal scores are resolved deterministically by the original input index.
    """
    boxes_array = _as_boxes(boxes, "boxes")
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 1 or len(scores_array) != len(boxes_array):
        raise ValueError("scores must be a 1D array with one score per box")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if len(boxes_array) == 0:
        return np.empty((0,), dtype=np.int64)

    # lexsort uses the last key as primary: score descending, then index ascending.
    order = np.lexsort((np.arange(len(scores_array)), -scores_array))
    kept: list[int] = []

    while order.size:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if remaining.size == 0:
            break
        overlaps = pairwise_iou(boxes_array[current:current + 1], boxes_array[remaining])[0]
        # Exact-threshold overlaps are kept; only IoU > threshold is suppressed.
        order = remaining[overlaps <= iou_threshold]

    return np.asarray(kept, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray | Sequence[Sequence[float]],
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """NMS independently per class; return original indices, score-sorted globally."""
    boxes_array = _as_boxes(boxes, "boxes")
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels)
    if scores_array.ndim != 1 or labels_array.ndim != 1:
        raise ValueError("scores and labels must be 1D arrays")
    if len(boxes_array) != len(scores_array) or len(boxes_array) != len(labels_array):
        raise ValueError("boxes, scores and labels must have the same length")
    if len(boxes_array) == 0:
        return np.empty((0,), dtype=np.int64)

    kept: list[int] = []
    for label in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == label)
        local_kept = nms(boxes_array[class_indices], scores_array[class_indices], iou_threshold)
        kept.extend(class_indices[local_kept].tolist())

    # Deterministic global ordering: descending score, then original index.
    kept_array = np.asarray(kept, dtype=np.int64)
    global_order = np.lexsort((kept_array, -scores_array[kept_array]))
    return kept_array[global_order]


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-points interpolated AP (VOC 2010+ style precision envelope)."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    change_points = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1]))


def evaluate_detections(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    class_ids: Iterable[int],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute per-class AP and macro mAP for one list of images.

    Each predictions[i] must contain:
      boxes: (N, 4) xyxy array
      labels: (N,) integer class IDs
      scores: (N,) confidence scores

    Each targets[i] must contain:
      boxes: (M, 4) xyxy array
      labels: (M,) integer class IDs

    A prediction is a TP if it matches an as-yet-unmatched ground-truth box of
    the same class in the same image with IoU >= iou_threshold. Duplicate matches
    are FPs. Classes with no ground-truth instances receive AP=None and are
    excluded from macro mAP. AP uses all-points interpolated precision-envelope
    integration (VOC 2010+ style).
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of images")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    normalized_predictions = []
    normalized_targets = []
    discovered_classes: set[int] = set()

    for index, (pred, target) in enumerate(zip(predictions, targets)):
        pred_boxes = _as_boxes(pred.get("boxes", []), f"predictions[{index}].boxes")
        pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64)
        pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64)
        target_boxes = _as_boxes(target.get("boxes", []), f"targets[{index}].boxes")
        target_labels = np.asarray(target.get("labels", []), dtype=np.int64)

        if pred_labels.ndim != 1 or pred_scores.ndim != 1 or len(pred_boxes) != len(pred_labels) or len(pred_boxes) != len(pred_scores):
            raise ValueError(f"Invalid prediction lengths at image index {index}")
        if target_labels.ndim != 1 or len(target_boxes) != len(target_labels):
            raise ValueError(f"Invalid target lengths at image index {index}")
        if not np.isfinite(pred_scores).all():
            raise ValueError("prediction scores must be finite")

        discovered_classes.update(map(int, pred_labels.tolist()))
        discovered_classes.update(map(int, target_labels.tolist()))
        normalized_predictions.append((pred_boxes, pred_labels, pred_scores))
        normalized_targets.append((target_boxes, target_labels))

    requested_classes = list(dict.fromkeys(int(c) for c in class_ids))
    if not requested_classes:
        requested_classes = sorted(discovered_classes)

    per_class: dict[int, dict] = {}
    valid_aps: list[float] = []

    for class_id in requested_classes:
        gt_by_image: list[np.ndarray] = []
        matched_by_image: list[np.ndarray] = []
        num_ground_truth = 0
        all_detections: list[tuple[float, int, int, np.ndarray]] = []

        for image_index, ((pred_boxes, pred_labels, pred_scores), (gt_boxes, gt_labels)) in enumerate(
            zip(normalized_predictions, normalized_targets)
        ):
            gt_for_class = gt_boxes[gt_labels == class_id]
            gt_by_image.append(gt_for_class)
            matched_by_image.append(np.zeros(len(gt_for_class), dtype=bool))
            num_ground_truth += len(gt_for_class)

            pred_indices = np.flatnonzero(pred_labels == class_id)
            for pred_index in pred_indices:
                all_detections.append(
                    (float(pred_scores[pred_index]), image_index, int(pred_index), pred_boxes[pred_index])
                )

        if num_ground_truth == 0:
            per_class[class_id] = {
                "ap": None,
                "ground_truth_count": 0,
                "prediction_count": len(all_detections),
                "tp": 0,
                "fp": len(all_detections),
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        # Stable deterministic ordering: confidence descending, then image and box index.
        all_detections.sort(key=lambda item: (-item[0], item[1], item[2]))
        tp = np.zeros(len(all_detections), dtype=np.float64)
        fp = np.zeros(len(all_detections), dtype=np.float64)

        for detection_index, (_, image_index, _, pred_box) in enumerate(all_detections):
            gt_boxes = gt_by_image[image_index]
            if len(gt_boxes) == 0:
                fp[detection_index] = 1.0
                continue

            overlaps = pairwise_iou(pred_box.reshape(1, 4), gt_boxes)[0]
            best_gt_index = int(np.argmax(overlaps))
            best_iou = float(overlaps[best_gt_index])
            if best_iou >= iou_threshold and not matched_by_image[image_index][best_gt_index]:
                tp[detection_index] = 1.0
                matched_by_image[image_index][best_gt_index] = True
            else:
                fp[detection_index] = 1.0

        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall_curve = cumulative_tp / num_ground_truth
        precision_curve = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)
        ap = _average_precision(recall_curve, precision_curve)
        final_tp = int(cumulative_tp[-1]) if cumulative_tp.size else 0
        final_fp = int(cumulative_fp[-1]) if cumulative_fp.size else 0
        final_fn = num_ground_truth - final_tp
        final_precision = final_tp / (final_tp + final_fp) if final_tp + final_fp else 0.0
        final_recall = final_tp / num_ground_truth

        per_class[class_id] = {
            "ap": ap,
            "ground_truth_count": num_ground_truth,
            "prediction_count": len(all_detections),
            "tp": final_tp,
            "fp": final_fp,
            "fn": final_fn,
            "precision": final_precision,
            "recall": final_recall,
        }
        valid_aps.append(ap)

    return {
        "iou_threshold": float(iou_threshold),
        "ap_by_class": per_class,
        "map": float(np.mean(valid_aps)) if valid_aps else None,
        "classes_in_map": len(valid_aps),
        "ap_method": "all-points interpolated precision envelope (VOC 2010+)",
    }


def evaluate_map50(predictions: Sequence[dict], targets: Sequence[dict], class_ids: Iterable[int]) -> dict:
    """Convenience wrapper for mAP@0.5."""
    return evaluate_detections(predictions, targets, class_ids, iou_threshold=0.5)
