"""Run major CLI paths on synthetic data; resulting models are not useful for ALPR."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from alpr.io import new_run, write_json


def run(output: Path, architecture: str) -> None:
    output = new_run(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "YOLO_CONFIG_DIR": str(output / "yolo-config"),
        "MPLCONFIGDIR": str(output / "matplotlib"),
    }
    fixture = output / "raw"
    chars, detector = output / "characters", output / "detection"
    recognition_run, detection_run = output / "recognizer", output / "detector"
    completed = []

    def command(arguments: list[str], module: bool = True):
        cmd = [sys.executable, "-m", "alpr", *arguments] if module else [sys.executable, *arguments]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, cwd=repository, check=True)
        completed.append(arguments)

    command(
        [str(repository / "scripts/make_synthetic_fixture.py"), "--output", str(fixture)], False
    )
    command(["prepare-characters", "--source", str(fixture / "characters"), "--output", str(chars)])
    command(
        [
            "prepare-detection",
            "--images",
            str(fixture / "detection/images"),
            "--labels",
            str(fixture / "detection/labels"),
            "--output",
            str(detector),
        ]
    )
    command(
        [
            "train-recognizer",
            "--manifest",
            str(chars / "manifest.csv"),
            "--output",
            str(recognition_run),
            "--architecture",
            architecture,
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--no-pretrained",
            "--device",
            "cpu",
        ]
    )
    command(
        [
            "train-detector",
            "--data",
            str(detector / "data.yaml"),
            "--output",
            str(detection_run),
            "--model",
            "yolov8n.yaml",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "64",
            "--no-pretrained",
            "--device",
            "cpu",
        ]
    )
    command(
        [
            "evaluate-recognizer",
            "--manifest",
            str(chars / "manifest.csv"),
            "--weights",
            str(recognition_run / "best.pt"),
            "--output",
            str(output / "recognizer-test.json"),
            "--batch-size",
            "7",
            "--device",
            "cpu",
        ]
    )
    detector_weights = detection_run / "fit/weights/best.pt"
    command(
        [
            "evaluate-detector",
            "--data",
            str(detector / "data.yaml"),
            "--weights",
            str(detector_weights),
            "--output",
            str(output / "detector-test"),
            "--image-size",
            "64",
            "--device",
            "cpu",
        ]
    )
    common = [
        "--detector",
        str(detector_weights),
        "--recognizer",
        str(recognition_run / "best.pt"),
        "--device",
        "cpu",
        "--image-size",
        "64",
    ]
    command(
        [
            "infer",
            *common,
            "--source",
            str(fixture / "detection/images/000.png"),
            "--output",
            str(output / "inference"),
            "--debug",
        ]
    )
    command(
        [
            "evaluate",
            *common,
            "--manifest",
            str(fixture / "benchmark.jsonl"),
            "--output",
            str(output / "end-to-end"),
        ]
    )
    # Verify sample denominators instead of artificial accuracy targets.
    report = json.loads((output / "recognizer-test.json").read_text())
    assert report["samples"] == 36
    report = json.loads((output / "end-to-end/metrics.json").read_text())
    assert report["ground_truth_plates"] == 12 and report["images"] == 12
    write_json(
        output / "smoke-result.json",
        {
            "software_smoke_test": True,
            "research_accuracy_verified": False,
            "architecture": architecture,
            "completed_commands": completed,
        },
    )
    print(f"Software smoke test completed: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/software-smoke"))
    p.add_argument(
        "--architecture", choices=["efficientnet_b0", "efficientnet_b7"], default="efficientnet_b0"
    )
    args = p.parse_args()
    run(args.output, args.architecture)
