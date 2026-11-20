"""Train fresh models and evaluate the prepared public data in one recorded workflow."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from alpr.io import new_run, write_json
from alpr.recognition import resolve_device


def run(prepared: Path, output: Path, device: str, recognizer: Path | None) -> None:
    prepared = prepared.resolve()
    if recognizer is not None and not recognizer.is_file():
        raise ValueError("Provided recognizer checkpoint does not exist")
    for relative in ["mercosur/data.yaml", "characters/manifest.csv", "openalpr/benchmark.jsonl"]:
        if not (prepared / relative).is_file():
            raise ValueError(f"Prepare the public datasets first: missing {relative}")
    output = new_run(output).resolve()
    device = str(resolve_device(device))
    commands = []

    def command(args):
        print("Running:", " ".join(args), flush=True)
        subprocess.run([sys.executable, "-m", "alpr", *args], check=True)
        commands.append(args)
        write_json(
            output / "progress.json", {"completed_commands": commands, "status": "in_progress"}
        )

    detector_run, recognition_run = output / "detector", output / "recognizer"
    command(
        [
            "train-detector",
            "--data",
            str(prepared / "mercosur/data.yaml"),
            "--output",
            str(detector_run),
            "--epochs",
            "10",
            "--device",
            device,
        ]
    )
    if recognizer is None:
        command(
            [
                "train-recognizer",
                "--manifest",
                str(prepared / "characters/manifest.csv"),
                "--output",
                str(recognition_run),
                "--architecture",
                "efficientnet_b7",
                "--epochs",
                "15",
                "--batch-size",
                "64",
                "--device",
                device,
            ]
        )
        recognizer = recognition_run / "best.pt"
    detector = detector_run / "fit/weights/best.pt"
    command(
        [
            "evaluate-detector",
            "--data",
            str(prepared / "mercosur/data.yaml"),
            "--weights",
            str(detector),
            "--output",
            str(output / "detector-test"),
            "--device",
            device,
        ]
    )
    command(
        [
            "evaluate-recognizer",
            "--manifest",
            str(prepared / "characters/manifest.csv"),
            "--weights",
            str(recognizer),
            "--output",
            str(output / "character-test.json"),
            "--device",
            device,
        ]
    )
    command(
        [
            "evaluate",
            "--detector",
            str(detector),
            "--recognizer",
            str(recognizer),
            "--manifest",
            str(prepared / "openalpr/benchmark.jsonl"),
            "--output",
            str(output / "openalpr-test"),
            "--device",
            device,
        ]
    )
    first = json.loads((prepared / "openalpr/benchmark.jsonl").read_text().splitlines()[0])
    command(
        [
            "infer",
            "--detector",
            str(detector),
            "--recognizer",
            str(recognizer),
            "--source",
            str(prepared / "openalpr" / first["image"]),
            "--output",
            str(output / "example"),
            "--debug",
            "--device",
            device,
        ]
    )
    write_json(
        output / "progress.json",
        {
            "completed_commands": commands,
            "status": "complete",
            "original_paper_benchmark_recovered": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/public-experiment"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--recognizer-checkpoint",
        type=Path,
        help="Reuse a separately trained checkpoint; otherwise train B7",
    )
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path("artifacts/torch").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("artifacts/ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))
    run(args.prepared, args.output, args.device, args.recognizer_checkpoint)
