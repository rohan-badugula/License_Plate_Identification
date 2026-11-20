from pathlib import Path

import numpy as np
import pytest

from alpr.detection import PlateDetector, validate_detector_data


@pytest.mark.integration
def test_archived_detector_loads_and_predicts(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "yolo"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    path = Path(__file__).parents[1] / "Code/YOLO_V8_resluts/weights/best.pt"
    if not path.is_file():
        pytest.skip("Archived detector not included in this source distribution")
    detector = PlateDetector(path, image_size=64)
    assert len(detector.model.names) == 1
    result = detector.predict(np.zeros((64, 96, 3), np.uint8))
    assert isinstance(result, list)
    for prediction in result:
        assert len(prediction["box"]) == 4
        assert 0 <= prediction["confidence"] <= 1


def test_missing_or_invalid_detector_configuration(tmp_path):
    with pytest.raises(FileNotFoundError):
        PlateDetector(tmp_path / "missing.pt")


def test_yaml_cannot_bypass_the_audited_manifest(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text('{"path":"/tmp/other-data", "names":{"0":"license_plate"}}')
    with pytest.raises(ValueError, match="prepared"):
        validate_detector_data(path, "test")
