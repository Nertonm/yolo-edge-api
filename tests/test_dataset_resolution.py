from pathlib import Path

import yaml

from preprocessing.utils.evaluate import (
    resolve_dataset_yaml,
    transform_yolo_label_text,
)


def test_resolve_dataset_yaml_uses_repo_relative_paths():
    data_yaml = Path("dataset/exports/epi-v1/data.yaml")
    resolved = resolve_dataset_yaml(str(data_yaml))

    assert Path(resolved).exists()
    assert Path(resolved).is_absolute()

    cfg = yaml.safe_load(Path(resolved).read_text())
    assert cfg["path"] == str(data_yaml.parent.resolve())
    assert cfg["val"] == str((data_yaml.parent / "valid/images").resolve())
    assert cfg["test"] == str((data_yaml.parent / "test/images").resolve())


def test_transform_yolo_label_text_adjusts_letterbox_padding():
    # 640x480 letterboxed into 416x416: vertical padding is added.
    raw = "0 0.5 0.5 0.25 0.25\n"
    transformed = transform_yolo_label_text(
        raw, (480, 640), (416, 416), "letterbox"
    )
    values = [float(v) for v in transformed.split()]
    assert values[0] == 0
    assert values[1] == 0.5
    assert values[2] == 0.5
    assert values[3] == 0.25
    assert values[4] < 0.25
