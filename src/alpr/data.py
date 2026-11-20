"""Validated datasets, persisted splits, and explicit character vocabulary."""

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .io import IMAGE_SUFFIXES, new_run, read_image, sha256, write_json

VOCABULARY = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SPLITS = ("train", "val", "test")


def validate_yolo(path: Path) -> list[list[float]]:
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: nonnumeric annotation") from error
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{line_number}: expected 5 finite YOLO values")
        cls, x, y, w, h = values
        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{line_number}: expected class 0 and normalized xywh")
        if x - w / 2 < -1e-6 or y - h / 2 < -1e-6:
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        if x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        boxes.append(values)
    return boxes


def grouped_split(records: list[dict], val: float, test: float, seed: int) -> list[dict]:
    """Assign whole groups. Exact percentages may differ when groups have unequal sizes."""
    if not (0 < val < 1 and 0 < test < 1 and val + test < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Need at least three independent groups for train/val/test")
    random.Random(seed).shuffle(keys)
    # Allocate by group count, reserving at least one group for each split.
    n_val = max(1, round(len(keys) * val))
    n_test = max(1, round(len(keys) * test))
    if n_val + n_test >= len(keys):
        raise ValueError("Too few independent groups for the requested split fractions")
    for i, key in enumerate(keys):
        split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
        for record in groups[key]:
            record["split"] = split
    return records


def save_manifest(output: Path, records: list[dict], metadata: dict) -> Path:
    fields = sorted(set().union(*(record.keys() for record in records)))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    metadata["counts"] = {s: sum(r["split"] == s for r in records) for s in SPLITS}
    metadata["manifest_sha256"] = sha256(output / "manifest.csv")
    write_json(output / "dataset.json", metadata)
    return output / "manifest.csv"


def prepare_detection(
    images: Path,
    labels: Path,
    output: Path,
    *,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    groups_file: Path | None = None,
) -> Path:
    images, labels = images.resolve(), labels.resolve()
    if not images.is_dir() or not labels.is_dir():
        raise ValueError("Image and label directories must exist")
    groups = {}
    if groups_file:
        with groups_file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get("path") or not row.get("group") or row["path"] in groups:
                    raise ValueError("Groups CSV needs unique path and nonempty group columns")
                groups[row["path"]] = row["group"]
    records, label_hashes, duplicate_groups = [], {}, {}
    files = sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for image in files:
        relative = image.relative_to(images)
        label = labels / relative.with_suffix(".txt")
        if not label.is_file():
            raise ValueError(
                f"Missing label for {relative}; use an empty file for a negative image"
            )
        read_image(image)
        validate_yolo(label)
        digest, label_digest = sha256(image), sha256(label)
        if digest in label_hashes and label_hashes[digest] != label_digest:
            raise ValueError(f"Duplicate image has conflicting annotations: {relative}")
        label_hashes[digest] = label_digest
        group = groups.get(relative.as_posix(), digest)
        if groups_file and relative.as_posix() not in groups:
            raise ValueError(f"Missing group for {relative}")
        if digest in duplicate_groups and duplicate_groups[digest] != group:
            raise ValueError(f"Duplicate image assigned to different groups: {relative}")
        duplicate_groups[digest] = group
        records.append(
            {
                "source": relative.as_posix(),
                "sha256": digest,
                "label_sha256": label_digest,
                "group": group,
            }
        )
    grouped_split(records, val, test, seed)
    # Validate before creating output. Never silently merge with an old prepared dataset.
    output = new_run(output).resolve()
    destinations = set()
    for record in records:
        relative = Path(record["source"])
        # Keep subdirectory identity without requiring matching basenames to be globally unique.
        name = relative.as_posix().replace("/", "__")
        target = output / "images" / record["split"] / name
        label_target = output / "labels" / record["split"] / Path(name).with_suffix(".txt")
        if label_target in destinations:
            raise ValueError(f"Prepared label filename collision: {relative}")
        destinations.add(label_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images / relative, target)
        shutil.copy2(labels / relative.with_suffix(".txt"), label_target)
        record["path"] = target.relative_to(output).as_posix()
        record["label_path"] = label_target.relative_to(output).as_posix()
    # JSON is valid YAML, with no extra YAML parser required for generation.
    write_json(
        output / "data.yaml",
        {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "license_plate"},
        },
    )
    return save_manifest(
        output,
        records,
        {"kind": "detection", "seed": seed, "val_fraction": val, "test_fraction": test},
    )


def prepare_characters(
    source: Path,
    output: Path,
    *,
    mapping_file: Path | None = None,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Path:
    source = source.resolve()
    mapping = json.loads(mapping_file.read_text()) if mapping_file else {c: c for c in VOCABULARY}
    if not isinstance(mapping, dict) or set(mapping.values()) != set(VOCABULARY):
        raise ValueError("Mapping must cover all 36 alphanumeric characters exactly once")
    if len(mapping) != 36:
        raise ValueError("Mapping must contain exactly 36 folders")
    records, seen = [], {}
    for folder, character in mapping.items():
        directory = source / folder
        if not directory.is_dir():
            raise ValueError(f"Missing character folder {directory}")
        class_records = []
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            read_image(path)
            digest = sha256(path)
            if digest in seen and seen[digest] != character:
                raise ValueError(f"Same image assigned different character labels: {path}")
            seen[digest] = character
            class_records.append(
                {
                    "source": path.relative_to(source).as_posix(),
                    "label": character,
                    "sha256": digest,
                    "group": digest,
                }
            )
        grouped_split(class_records, val, test, seed)
        records.extend(class_records)
    output = new_run(output).resolve()
    for record in records:
        relative = Path(record["source"])
        target = output / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        record["path"] = target.relative_to(output).as_posix()
    return save_manifest(
        output,
        records,
        {
            "kind": "characters",
            "vocabulary": VOCABULARY,
            "seed": seed,
            "val_fraction": val,
            "test_fraction": test,
            "folder_mapping": mapping,
        },
    )


def load_manifest(path: Path, split: str | None = None, verify_hashes: bool = True) -> list[dict]:
    """Check all splits for leakage, even when reading only the training partition."""
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Manifest is empty")
    assignments = {}
    root = path.parent.resolve()
    for record in records:
        if record.get("split") not in SPLITS:
            raise ValueError("Manifest split must be train, val or test")
        for key in (record.get("group"), record.get("sha256")):
            if not key:
                raise ValueError("Manifest requires nonempty group and sha256 fields")
            if key in assignments and assignments[key] != record["split"]:
                raise ValueError(
                    "Data leakage: group or identical image appears in multiple splits"
                )
            assignments[key] = record["split"]
        for field, hash_field in (("path", "sha256"), ("label_path", "label_sha256")):
            if field not in record:
                continue
            item = (root / record[field]).resolve()
            if not item.is_relative_to(root):
                raise ValueError("Manifest paths must remain inside the dataset directory")
            if not item.is_file():
                raise ValueError(f"Missing manifest file: {item}")
            if verify_hashes and sha256(item) != record[hash_field]:
                raise ValueError(f"Dataset content changed since preparation: {item}")
    selected = [r for r in records if split is None or r["split"] == split]
    if not selected:
        raise ValueError(f"No samples in split {split}")
    return selected
