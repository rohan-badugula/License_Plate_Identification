"""Compose detector, segmentation and classifier without an OCR substitute."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .data import VOCABULARY
from .io import IMAGE_SUFFIXES, environment, new_run, read_image, sha256, write_image, write_json
from .metrics import normalize_text, plate_metrics, validate_box
from .segmentation import SegmentationConfig, segment


class ALPRPipeline:
    def __init__(
        self,
        detector,
        recognizer,
        segmentation: SegmentationConfig | None = None,
        min_character_confidence: float = 0.0,
    ):
        if not 0 <= min_character_confidence <= 1:
            raise ValueError("Character confidence threshold must be in [0, 1]")
        self.detector, self.recognizer = detector, recognizer
        self.segmentation = segmentation or SegmentationConfig()
        self.min_character_confidence = min_character_confidence

    def predict(self, image: np.ndarray, debug_directory: Path | None = None) -> dict:
        started = time.perf_counter()
        height, width = image.shape[:2]
        plates = []
        for detection in self.detector.predict(image):
            coordinates = detection["box"]
            if len(coordinates) != 4 or not all(math.isfinite(v) for v in coordinates):
                continue
            x1 = max(0, min(width, math.floor(coordinates[0])))
            y1 = max(0, min(height, math.floor(coordinates[1])))
            x2 = max(0, min(width, math.ceil(coordinates[2])))
            y2 = max(0, min(height, math.ceil(coordinates[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            glyphs, debug = segment(crop, self.segmentation)
            predictions = self.recognizer.predict([g.image for g in glyphs])
            if len(predictions) != len(glyphs):
                raise ValueError("Recognizer returned a different number of glyph predictions")
            characters = []
            for glyph, prediction in zip(glyphs, predictions, strict=True):
                gx1, gy1, gx2, gy2 = glyph.box
                box = [
                    x1 + gx1 * (x2 - x1) / self.segmentation.width,
                    y1 + gy1 * (y2 - y1) / self.segmentation.height,
                    x1 + gx2 * (x2 - x1) / self.segmentation.width,
                    y1 + gy2 * (y2 - y1) / self.segmentation.height,
                ]
                characters.append({**prediction, "box": box})
            text = "".join(c["character"] for c in characters)
            confidence = min((c["confidence"] for c in characters), default=0.0)
            status = (
                "no_characters"
                if not characters
                else "low_confidence"
                if confidence < self.min_character_confidence
                else "recognized"
            )
            plates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": detection["confidence"],
                    "text": text,
                    "recognition_confidence": confidence,
                    "status": status,
                    "characters": characters,
                }
            )
            if debug_directory:
                directory = debug_directory / f"plate_{len(plates) - 1:03d}"
                write_image(directory / "crop.png", crop)
                write_image(directory / "threshold.png", debug["mask"])
                for index, glyph in enumerate(glyphs):
                    write_image(directory / f"character_{index:03d}.png", glyph.image)
        return {
            "width": width,
            "height": height,
            "plates": plates,
            "elapsed_seconds": time.perf_counter() - started,
        }


def annotate(image: np.ndarray, result: dict) -> np.ndarray:
    canvas = image.copy()
    for plate in result["plates"]:
        x1, y1, x2, y2 = plate["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.putText(
            canvas,
            plate["text"] or plate["status"],
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 210, 0),
            2,
        )
    return canvas


def pipeline_provenance(pipeline: ALPRPipeline) -> dict:
    return {
        "environment": environment(),
        "segmentation": asdict(pipeline.segmentation),
        "min_character_confidence": pipeline.min_character_confidence,
        "recognition": asdict(pipeline.recognizer.config),
        "recognition_training_status": pipeline.recognizer.metadata.get(
            "training_status", "unknown"
        ),
        "recognizer_device": str(pipeline.recognizer.device),
        "confidence_definition": (
            "plate confidence is detector score; recognition is minimum glyph score"
        ),
    }


def infer_images(
    pipeline: ALPRPipeline, source: Path, output: Path, *, debug: bool = False
) -> dict:
    paths = (
        sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if source.is_dir()
        else [source]
    )
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Source must contain at least one readable image")
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the input directory to avoid recursive inference")
    output = new_run(output)
    results = []
    for index, path in enumerate(paths):
        image = read_image(path)
        key = f"{index:05d}_{path.stem}"
        result = pipeline.predict(image, output / "debug" / key if debug else None)
        result.update(source=str(path), source_sha256=sha256(path))
        results.append(result)
        write_image(output / "annotated" / f"{key}.jpg", annotate(image, result))
    summary = {**pipeline_provenance(pipeline), "images": results}
    write_json(output / "predictions.json", summary)
    return summary


def infer_video(pipeline: ALPRPipeline, source: Path, output: Path, *, stride: int = 1) -> dict:
    if stride < 1 or not source.is_file():
        raise ValueError("Video must exist and stride must be positive")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    output = new_run(output)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video has no valid frame rate")
    writer = None
    processed = frames_read = 0
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = frames_read
                frames_read += 1
                if index % stride:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / stride,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise ValueError("Cannot create MP4 output with the mp4v codec")
                result = pipeline.predict(frame)
                result.update(frame_index=index, timestamp_seconds=index / fps)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                writer.write(annotate(frame, result))
                processed += 1
    finally:
        capture.release()
        if writer:
            writer.release()
    if not processed:
        raise ValueError("Video contains no decodable frames")
    summary = {
        **pipeline_provenance(pipeline),
        "source_sha256": sha256(source),
        "source": str(source),
        "input_fps": fps,
        "stride": stride,
        "frames_read": frames_read,
        "frames_processed": processed,
        "output_fps": fps / stride,
        "audio_preserved": False,
        "tracking": False,
    }
    write_json(output / "video.json", summary)
    return summary


def evaluate_pipeline(
    pipeline: ALPRPipeline, manifest: Path, output: Path, *, iou_threshold: float = 0.5
) -> dict:
    """Full images with pixel-coordinate boxes and text, one JSON object per line."""
    samples, seen, seen_hashes = [], set(), set()
    with manifest.open(encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream if line.strip()]
    if not entries:
        raise ValueError("End-to-end evaluation manifest is empty")
    for entry in entries:
        path = (manifest.parent / entry["image"]).resolve()
        if path in seen:
            raise ValueError("Duplicate evaluation image")
        seen.add(path)
        image = read_image(path)
        digest = sha256(path)
        if digest in seen_hashes:
            raise ValueError("Duplicate evaluation image content")
        seen_hashes.add(digest)
        height, width = image.shape[:2]
        truth = entry["plates"]
        for plate in truth:
            validate_box(plate["box"])
            if plate["box"][2] > width or plate["box"][3] > height:
                raise ValueError("Ground-truth plate box exceeds image dimensions")
            text = normalize_text(plate["text"])
            if not text or any(c not in VOCABULARY for c in text):
                raise ValueError("Ground-truth text must contain supported alphanumeric characters")
        result = pipeline.predict(image)
        samples.append(
            {
                "image": entry["image"],
                "source_sha256": digest,
                "truth": truth,
                "predictions": result["plates"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    output = new_run(output)
    summary = {
        **pipeline_provenance(pipeline),
        **plate_metrics(samples, iou_threshold),
        "manifest_sha256": sha256(manifest),
        "latency_includes": "detector, segmentation and recognition; excludes file I/O",
        "mean_latency_seconds": float(np.mean([s["elapsed_seconds"] for s in samples])),
        "samples": samples,
    }
    write_json(output / "metrics.json", summary)
    return summary
