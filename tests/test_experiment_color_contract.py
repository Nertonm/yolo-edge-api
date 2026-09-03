import numpy as np

from preprocessing.experiments.e3_filters import (
    preproc_gauss_33,
    preproc_no_filter,
)
from preprocessing.experiments.e4_contrast import (
    clahe_hsv,
    equalize_hist_hsv,
    no_equalization,
)


def make_bgr_frame():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 80
    frame[..., 2] = 200
    return frame


def test_e3_no_filter_preserves_bgr_for_serialized_evaluation():
    frame = make_bgr_frame()
    np.testing.assert_array_equal(preproc_no_filter(frame), frame)


def test_e3_blur_preserves_bgr_channel_order():
    result = preproc_gauss_33(make_bgr_frame())
    assert result[0, 0, 0] == 10
    assert result[0, 0, 1] == 80
    assert result[0, 0, 2] == 200


def test_e4_contrast_variants_return_bgr_arrays():
    frame = make_bgr_frame()
    np.testing.assert_array_equal(no_equalization(frame), frame)
    for transform in [equalize_hist_hsv, clahe_hsv]:
        result = transform(frame)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255
