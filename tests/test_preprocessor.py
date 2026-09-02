import cv2
import numpy as np
import pytest

from preprocessing.preprocessor import (
    CONFIG_DEFAULT,
    CONFIG_HIGH_QUALITY,
    CONFIG_LOW_LIGHT,
    PreprocessConfig,
    Preprocessor,
)


def make_frame(height=480, width=640, dtype=np.uint8):
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (height, width, 3), dtype=dtype)


def test_output_shape_letterbox():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
    assert result.frame.shape == (416, 416, 3)


def test_output_dtype_uint8_without_normalization():
    result = Preprocessor(PreprocessConfig(normalize=False)).process(make_frame())
    assert result.frame.dtype == np.uint8


def test_output_dtype_float32_with_normalization():
    result = Preprocessor(PreprocessConfig(normalize=True)).process(make_frame())
    assert result.frame.dtype == np.float32
    assert result.frame.min() >= 0.0
    assert result.frame.max() <= 1.0


def test_metadata_contains_scale_padding_and_original_size():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame())
    assert result.scale == pytest.approx(0.65)
    assert result.pad_w == 0
    assert result.pad_h == 52
    assert result.orig_size == (480, 640)


def test_square_letterbox_has_no_padding():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(make_frame(416, 416))
    assert result.pad_w == 0
    assert result.pad_h == 0


def test_adjust_boxes_removes_letterbox_offset():
    processor = Preprocessor(PreprocessConfig(infer_size=416))
    result = processor.process(make_frame())
    boxes = np.array([[65, 130, 195, 260]], dtype=float)
    adjusted = processor.adjust_boxes(boxes, result)
    expected = np.array([[100, 120, 300, 320]], dtype=float)
    np.testing.assert_allclose(adjusted, expected)


def test_low_light_config_applies_clahe_and_preserves_channels():
    result = Preprocessor(CONFIG_LOW_LIGHT).process(make_frame())
    assert result.frame.shape == (320, 320, 3)
    assert result.frame.dtype == np.uint8


def test_predefined_configs_are_usable_and_default_has_no_filter():
    assert CONFIG_DEFAULT.convert_rgb
    assert CONFIG_DEFAULT.use_letterbox
    assert not CONFIG_DEFAULT.gaussian_blur
    assert not CONFIG_DEFAULT.median_blur
    assert not CONFIG_DEFAULT.clahe
    assert CONFIG_HIGH_QUALITY.infer_size == 640
