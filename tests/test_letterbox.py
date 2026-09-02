import numpy as np

from preprocessing.utils.letterbox import adjust_bboxes, letterbox


def test_letterbox_preserves_shape_and_reports_padding():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, scale, (pad_w, pad_h) = letterbox(frame, target_size=416)
    assert result.shape == (416, 416, 3)
    assert scale == 0.65
    assert pad_w == 0
    assert pad_h == 52


def test_adjust_bboxes_reverses_letterbox_transform():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, scale, (pad_w, pad_h) = letterbox(frame, target_size=416)
    original = np.array([[100, 120, 300, 360]], dtype=float)
    boxed = original * scale
    boxed[:, [1, 3]] += pad_h
    restored = adjust_bboxes(boxed, scale, pad_w, pad_h)
    np.testing.assert_allclose(restored, original)
