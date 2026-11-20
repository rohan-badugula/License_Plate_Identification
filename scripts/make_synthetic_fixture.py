"""Generate tiny software fixtures, never a research benchmark or useful trained model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from alpr.data import VOCABULARY
from alpr.io import new_run, write_image, write_json


def create(output: Path) -> None:
    output = new_run(output)
    rng = np.random.default_rng(42)
    for character in VOCABULARY:
        for i in range(5):
            canvas = np.zeros((44, 44, 3), np.uint8)
            cv2.putText(
                canvas,
                character,
                (5 + i % 3, 33 + i % 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9 + i * 0.03,
                (255, 255, 255),
                1 + i % 2,
                cv2.LINE_AA,
            )
            canvas[0, 0] = [i + 1, ord(character), 3]  # Distinct source bytes.
            write_image(output / "characters" / character / f"{i}.png", canvas)
    entries = []
    for i in range(12):
        image = rng.integers(30, 100, (240, 480, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 60), (450, 220), (120 + i, 110, 80), -1)
        x, y, width, height = 90, 130, 300, 70
        image[y : y + height, x : x + width] = 255
        text = f"AB{i:02d}"
        for j, character in enumerate(text):
            cv2.putText(
                image,
                character,
                (x + 25 + j * 60, y + 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
        write_image(output / "detection" / "images" / f"{i:03d}.png", image)
        label = output / "detection" / "labels" / f"{i:03d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            f"0 {(x + width / 2) / 480} {(y + height / 2) / 240} {width / 480} {height / 240}\n"
        )
        entries.append(
            {
                "image": f"detection/images/{i:03d}.png",
                "plates": [{"box": [x, y, x + width, y + height], "text": text}],
            }
        )
    (output / "benchmark.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    write_json(
        output / "fixture.json",
        {"synthetic": True, "seed": 42, "purpose": "software smoke tests only"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args().output)
