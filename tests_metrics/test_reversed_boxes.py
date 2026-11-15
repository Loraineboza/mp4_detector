import numpy as np

from detection_metrics import box_iou, pairwise_iou, nms


def test_reversed_xyxy_coordinates_are_degenerate_not_reordered():
    normal = [0.0, 0.0, 10.0, 10.0]
    reversed_x = [10.0, 0.0, 0.0, 10.0]
    reversed_y = [0.0, 10.0, 10.0, 0.0]
    reversed_both = [10.0, 10.0, 0.0, 0.0]

    assert box_iou(reversed_x, normal) == 0.0
    assert box_iou(reversed_y, normal) == 0.0
    assert box_iou(reversed_both, normal) == 0.0


def test_pairwise_iou_returns_zero_for_reversed_and_degenerate_boxes():
    boxes_a = np.array([
        [10.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 10.0],
    ])
    boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_array_equal(pairwise_iou(boxes_a, boxes_b), np.zeros((2, 1)))


def test_nms_does_not_normalize_reversed_coordinates():
    boxes = np.array([
        [10.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 10.0],
    ])
    kept = nms(boxes, np.array([0.9, 0.8]), iou_threshold=0.5)
    np.testing.assert_array_equal(kept, np.array([0, 1]))
