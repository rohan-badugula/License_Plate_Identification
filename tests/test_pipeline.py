import json
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from alpr.pipeline import ALPRPipeline, evaluate_pipeline, infer_images, infer_video
from alpr.segmentation import SegmentationConfig


@dataclass
class FakeConfig:
    vocabulary: str = "AB08"


class FakeRecognizer:
    config = FakeConfig()
    device = "cpu"
    metadata = {"training_status": "test double"}

    def predict(self, glyphs):
        return [{"character": "AB08"[i % 4], "confidence": 0.9} for i in range(len(glyphs))]


class FakeDetector:
    def predict(self, image):
        h, w = image.shape[:2]
        return [
            {"box": [-5, -5, w + 5, h + 5], "confidence": 0.95},
            {"box": [0, 0, 0, 0], "confidence": 0.8},
        ]


def plate():
    image = np.full((75, 333, 3), 255, np.uint8)
    for i, character in enumerate("AB08"):
        cv2.putText(
            image, character, (30 + i * 60, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3
        )
    return image


def test_pipeline_clips_boxes_preserves_characters_and_debug(tmp_path):
    pipeline = ALPRPipeline(FakeDetector(), FakeRecognizer(), SegmentationConfig())
    result = pipeline.predict(plate(), tmp_path)
    assert len(result["plates"]) == 1
    prediction = result["plates"][0]
    assert prediction["box"] == [0, 0, 333, 75]
    assert prediction["text"] == "AB08"
    assert prediction["status"] == "recognized"
    assert len(list(tmp_path.rglob("character_*.png"))) == 4
    assert all(0 <= c["box"][0] < c["box"][2] <= 333 for c in prediction["characters"])


def test_no_plate_and_no_characters_are_valid():
    class EmptyDetector:
        def predict(self, image):
            return []

    assert ALPRPipeline(EmptyDetector(), FakeRecognizer()).predict(plate())["plates"] == []
    blank = np.full((75, 333, 3), 255, np.uint8)
    result = ALPRPipeline(FakeDetector(), FakeRecognizer()).predict(blank)
    assert result["plates"][0]["status"] == "no_characters"


def test_full_image_evaluation_and_image_output(tmp_path):
    image = tmp_path / "input.png"
    cv2.imwrite(str(image), plate())
    manifest = tmp_path / "truth.jsonl"
    manifest.write_text(
        json.dumps({"image": "input.png", "plates": [{"box": [0, 0, 333, 75], "text": "AB08"}]})
        + "\n"
    )
    pipeline = ALPRPipeline(FakeDetector(), FakeRecognizer())
    result = evaluate_pipeline(pipeline, manifest, tmp_path / "eval")
    assert result["plate_exact_match_rate"] == result["image_exact_match_rate"] == 1
    assert result["character_error_rate"] == 0
    infer_images(pipeline, image, tmp_path / "infer", debug=True)
    assert (tmp_path / "infer/predictions.json").is_file()
    assert list((tmp_path / "infer/annotated").glob("*.jpg"))
    with pytest.raises(ValueError, match="new or empty"):
        infer_images(pipeline, image, tmp_path / "infer")


def test_video_stream_timestamps_and_stride(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (334, 76))
    if not writer.isOpened():
        pytest.skip("mp4v codec is unavailable")
    frame = cv2.resize(plate(), (334, 76))
    for _ in range(5):
        writer.write(frame)
    writer.release()
    summary = infer_video(
        ALPRPipeline(FakeDetector(), FakeRecognizer()), path, tmp_path / "video", stride=2
    )
    assert summary["frames_processed"] == 3
    records = [
        json.loads(line) for line in (tmp_path / "video/predictions.jsonl").read_text().splitlines()
    ]
    assert [r["frame_index"] for r in records] == [0, 2, 4]
    assert [r["timestamp_seconds"] for r in records] == [0, 0.2, 0.4]
