import numpy as np
import pytest

from detection_metrics import (
    box_iou,
    pairwise_iou,
    nms,
    class_aware_nms,
    evaluate_detections,
)


def test_iou_identical_disjoint_and_partial():
    assert box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == pytest.approx(0.0)
    # Intersection=1, union=7.
    assert box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_pairwise_iou_shape_and_values():
    a = np.array([[0, 0, 2, 2], [5, 5, 6, 6]], dtype=float)
    b = np.array([[0, 0, 2, 2], [1, 1, 3, 3], [8, 8, 9, 9]], dtype=float)
    result = pairwise_iou(a, b)
    assert result.shape == (2, 3)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(1 / 7)
    assert result[1, 2] == pytest.approx(0.0)


def test_pairwise_iou_empty_inputs():
    assert pairwise_iou([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert pairwise_iou([[0, 0, 1, 1]], []).shape == (1, 0)
    assert pairwise_iou([], []).shape == (0, 0)


def test_nms_keeps_highest_score_and_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), [0, 2])


def test_nms_equal_scores_is_deterministic():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    np.testing.assert_array_equal(nms(boxes, [0.8, 0.8], 0.5), [0])


def test_nms_exact_threshold_is_not_suppressed():
    # Two 3x1 boxes offset by 1: intersection=2, union=3+3-2=4, IoU=0.5.
    boxes = np.array([[0, 0, 3, 1], [1, 0, 4, 1]], dtype=float)
    assert box_iou(boxes[0], boxes[1]) == pytest.approx(0.5)
    np.testing.assert_array_equal(nms(boxes, [0.9, 0.8], 0.5), [0, 1])


def test_class_aware_nms_does_not_suppress_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    kept = class_aware_nms(boxes, [0.9, 0.8], [1, 2], 0.5)
    np.testing.assert_array_equal(kept, [0, 1])


def test_metrics_known_tp_fp_fn_and_duplicate():
    # One GT object. First prediction matches; second is a duplicate and becomes FP.
    preds = [{
        "boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
        "labels": np.array([1, 1]),
        "scores": np.array([0.9, 0.8]),
    }]
    targets = [{"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([1])}]
    result = evaluate_detections(preds, targets, class_ids=[1])
    stats = result["ap_by_class"][1]
    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 0
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["ap"] == pytest.approx(1.0)
    assert result["map"] == pytest.approx(1.0)


def test_false_positive_when_no_gt_for_image_but_class_exists_elsewhere():
    preds = [
        {"boxes": np.array([[0, 0, 2, 2]]), "labels": np.array([1]), "scores": np.array([0.99])},
        {"boxes": np.empty((0, 4)), "labels": np.array([], dtype=int), "scores": np.array([])},
    ]
    targets = [
        {"boxes": np.empty((0, 4)), "labels": np.array([], dtype=int)},
        {"boxes": np.array([[10, 10, 12, 12]]), "labels": np.array([1])},
    ]
    result = evaluate_detections(preds, targets, class_ids=[1])
    stats = result["ap_by_class"][1]
    assert stats["tp"] == 0
    assert stats["fp"] == 1
    assert stats["fn"] == 1
    assert stats["ap"] == pytest.approx(0.0)


def test_class_without_ground_truth_is_excluded_from_map():
    preds = [{"boxes": np.array([[0, 0, 2, 2]]), "labels": np.array([2]), "scores": np.array([0.9])}]
    targets = [{"boxes": np.empty((0, 4)), "labels": np.array([], dtype=int)}]
    result = evaluate_detections(preds, targets, class_ids=[1, 2])
    assert result["ap_by_class"][1]["ap"] is None
    assert result["map"] is None


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        pairwise_iou([[0, 1, 2]], [[0, 0, 1, 1]])
    with pytest.raises(ValueError):
        nms([[0, 0, 1, 1]], [0.2, 0.3])
