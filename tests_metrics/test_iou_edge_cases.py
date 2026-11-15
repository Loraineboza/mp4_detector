import numpy as np
import pytest

from detection_metrics import box_iou, pairwise_iou, nms


def test_edge_touching_boxes_have_zero_iou():
    assert box_iou([0, 0, 2, 2], [2, 0, 4, 2]) == pytest.approx(0.0)


def test_corner_touching_boxes_have_zero_iou():
    assert box_iou([0, 0, 2, 2], [2, 2, 4, 4]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "degenerate",
    [
        [1, 1, 1, 3],
        [1, 2, 3, 2],
        [1, 1, 1, 1],
    ],
)
def test_zero_area_box_has_zero_finite_iou(degenerate):
    result = box_iou(degenerate, [0, 0, 4, 4])
    assert result == 0.0
    assert np.isfinite(result)


def test_two_zero_area_boxes_do_not_produce_nan():
    result = box_iou([1, 1, 1, 1], [1, 1, 1, 1])
    assert result == 0.0
    assert np.isfinite(result)


def test_pairwise_iou_handles_touching_and_zero_area_boxes():
    boxes_a = np.array([
        [0, 0, 2, 2],
        [1, 1, 1, 3],
    ], dtype=np.float64)
    boxes_b = np.array([
        [2, 0, 4, 2],
        [0, 0, 2, 2],
    ], dtype=np.float64)

    result = pairwise_iou(boxes_a, boxes_b)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[0.0, 1.0], [0.0, 0.0]])
    assert np.isfinite(result).all()


def test_nms_keeps_duplicate_zero_area_boxes():
    boxes = np.array([
        [1, 1, 1, 3],
        [1, 1, 1, 3],
    ], dtype=np.float64)
    kept = nms(boxes, [0.9, 0.8], iou_threshold=0.5)
    np.testing.assert_array_equal(kept, [0, 1])
