"""Locate downloaded datasets and prepare audited copies with source provenance."""

import argparse
import json
import shutil
from pathlib import Path

from alpr.benchmark import prepare_openalpr
from alpr.data import prepare_characters, prepare_detection
from alpr.io import write_json


def prepare(downloads: Path, output: Path, dataset: str, quarantine: bool) -> None:
    keys = ["mercosur", "characters", "openalpr"] if dataset == "all" else [dataset]
    for key in keys:
        source = downloads / key / "files"
        if not (downloads / key / "source.json").is_file():
            raise ValueError(f"Run download_datasets.py --dataset {key} first")
        if key == "mercosur":
            candidates = [p for p in source.rglob("images") if (p.parent / "labels").is_dir()]
            if len(candidates) != 1:
                raise ValueError("Expected one Mercosur images/labels root")
            images = candidates[0]
            labels = images.parent / "labels"
            prepare_detection(images, labels, output / key)
            image_stems = {p.stem for p in images.iterdir() if p.is_file()}
            orphans = [p.name for p in sorted(labels.glob("*.txt")) if p.stem not in image_stems]
            write_json(
                output / key / "source-audit.json",
                {
                    "orphan_labels": orphans,
                    "grouping": "byte-identical files only",
                    "vehicle_independence_verified": False,
                },
            )
        elif key == "characters":
            roots = list(source.rglob("dataset_characters"))
            if len(roots) != 1:
                raise ValueError("Expected one dataset_characters root")
            prepare_characters(roots[0], output / key, quarantine_conflicts=quarantine)
        else:
            roots = list(source.glob("*/endtoend"))
            if len(roots) != 1:
                raise ValueError("Expected one OpenALPR endtoend root")
            summary = prepare_openalpr(roots[0], output / key)
            print(json.dumps({k: v for k, v in summary.items() if k != "source_hashes"}, indent=2))
        shutil.copy2(downloads / key / "source.json", output / key / "source.json")
        print(f"Prepared {key}: {output / key}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downloads", type=Path, default=Path("data/downloads"))
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    p.add_argument(
        "--dataset", choices=["all", "mercosur", "characters", "openalpr"], default="all"
    )
    p.add_argument("--quarantine-conflicts", action="store_true")
    args = p.parse_args()
    prepare(args.downloads, args.output, args.dataset, args.quarantine_conflicts)
