from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_box_array(boxes):
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 4)
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError(f"Expected a box with 4 coordinates, got shape {arr.shape}")
        return arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Boxes must have shape (N, 4), got {arr.shape}")
    return arr


def compute_iou(box_a, box_b):
    """IoU for a single pair of boxes in [x1, y1, x2, y2] format."""
    box_a = _as_box_array(box_a)
    box_b = _as_box_array(box_b)
    if box_a.shape != (1, 4) or box_b.shape != (1, 4):
        raise ValueError("compute_iou expects a single box for each argument")
    return compute_iou_matrix(box_a, box_b)[0, 0]


def compute_iou_matrix(boxes_a, boxes_b):
    """Vectorized IoU matrix of shape (N, M) for all pairs of boxes."""
    boxes_a = _as_box_array(boxes_a)
    boxes_b = _as_box_array(boxes_b)

    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    x1 = np.maximum(boxes_a[:, 0][:, None], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, 1][:, None], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, 2][:, None], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, 3][:, None], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter_area

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.divide(
            inter_area,
            union,
            out=np.zeros_like(inter_area, dtype=np.float64),
            where=union > 0.0,
        )
    return iou


def non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.0):
    """Classic NMS in xyxy format with deterministic ordering for equal scores."""
    boxes = _as_box_array(boxes)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores length mismatch: {boxes.shape[0]} != {scores.shape[0]}")
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    valid = np.where(scores >= score_threshold)[0]
    if valid.size == 0:
        return np.array([], dtype=np.int64)

    boxes = boxes[valid]
    scores = scores[valid]
    order = np.argsort(-scores, kind="mergesort")

    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(valid[current])

        if order.size == 1:
            break

        remaining = order[1:]
        remaining_boxes = boxes[remaining]
        ious = compute_iou_matrix(np.asarray([boxes[current]], dtype=np.float64), remaining_boxes)[0]
        order = remaining[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def precision_recall(tp: Sequence[int], fp: Sequence[int], total_gt: int):
    """Return overall precision and recall for a detection list."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    if total_gt < 0:
        raise ValueError("total_gt must be non-negative")

    tp_total = float(tp.sum())
    fp_total = float(fp.sum())
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / float(total_gt) if total_gt > 0 else 0.0
    return precision, recall


def average_precision_from_tp_fp(tp: Sequence[int], fp: Sequence[int], total_gt: int):
    """11-point interpolated AP for a single class."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)

    if total_gt == 0:
        return 0.0

    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    precision = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, 1e-12)
    recall = cumsum_tp / float(total_gt)

    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_levels = np.linspace(0.0, 1.0, 11)
    ap = 0.0
    for level in recall_levels:
        values = precision[recall >= level]
        ap += (values.max() if values.size else 0.0) / len(recall_levels)
    return float(ap)


def evaluate_class_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """Evaluate one class, returning TP/FP/FN and AP for that class."""
    gt_boxes = _as_box_array(gt_boxes)
    pred_boxes = _as_box_array(pred_boxes)
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)

    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError("Number of gt boxes and gt labels mismatch")
    if pred_boxes.shape[0] != pred_labels.shape[0] or pred_boxes.shape[0] != pred_scores.shape[0]:
        raise ValueError("Number of pred boxes, labels and scores mismatch")

    class_id = set(gt_labels) | set(pred_labels)
    if len(class_id) == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "ap": 0.0}

    tp = []
    fp = []
    used_gt = set()
    gt_indices = np.where(gt_labels == next(iter(class_id)))[0]
    pred_indices = np.where(pred_labels == next(iter(class_id)))[0]

    for cls in sorted(class_id, key=lambda x: str(x)):
        gt_indices = np.where(gt_labels == cls)[0]
        pred_indices = np.where(pred_labels == cls)[0]
        if pred_indices.size == 0:
            total_gt = int(gt_indices.size)
            tp.extend([0] * total_gt)
            fp.extend([0] * total_gt)
            continue

        order = np.argsort(-pred_scores[pred_indices], kind="mergesort")
        sorted_pred_indices = pred_indices[order]
        local_tp = []
        local_fp = []
        matched_gt = set()

        for pred_idx in sorted_pred_indices:
            gt_candidates = np.array([gt_idx for gt_idx in gt_indices if gt_idx not in matched_gt], dtype=int)
            if gt_candidates.size == 0:
                local_tp.append(0)
                local_fp.append(1)
                continue

            ious = compute_iou_matrix(np.asarray([pred_boxes[pred_idx]], dtype=np.float64), gt_boxes[gt_candidates])[0]
            best_match = int(np.argmax(ious))
            best_iou = float(ious[best_match])
            if best_iou > iou_threshold:
                matched_gt.add(int(gt_candidates[best_match]))
                local_tp.append(1)
                local_fp.append(0)
            else:
                local_tp.append(0)
                local_fp.append(1)

        tp.extend(local_tp)
        fp.extend(local_fp)

    total_gt = int(gt_labels.shape[0])
    total_tp = int(np.sum(tp))
    total_fp = int(np.sum(fp))
    total_fn = max(total_gt - total_tp, 0)
    precision, recall = precision_recall(tp, fp, total_gt)
    ap = average_precision_from_tp_fp(tp, fp, total_gt)

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "ap": ap,
    }


def mean_average_precision(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """mAP@IoU with one AP value per class averaged across all classes."""
    gt_boxes = _as_box_array(gt_boxes)
    pred_boxes = _as_box_array(pred_boxes)
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)

    classes = sorted(set(gt_labels) | set(pred_labels), key=lambda x: str(x))
    ap_values = []
    for cls in classes:
        gt_cls = gt_boxes[gt_labels == cls]
        gt_cls_labels = gt_labels[gt_labels == cls]
        pred_cls = pred_boxes[pred_labels == cls]
        pred_cls_labels = pred_labels[pred_labels == cls]
        pred_cls_scores = pred_scores[pred_labels == cls]

        if pred_cls.size == 0 and gt_cls.size == 0:
            continue

        result = evaluate_class_predictions(
            gt_cls,
            gt_cls_labels,
            pred_cls,
            pred_cls_labels,
            pred_cls_scores,
            iou_threshold=iou_threshold,
        )
        ap_values.append(result["ap"])

    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


__all__ = [
    "compute_iou",
    "compute_iou_matrix",
    "non_max_suppression",
    "precision_recall",
    "average_precision_from_tp_fp",
    "evaluate_class_predictions",
    "mean_average_precision",
]
