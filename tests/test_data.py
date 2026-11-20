import csv
import json

import cv2
import numpy as np
import pytest

from alpr.data import (
    VOCABULARY,
    load_manifest,
    prepare_characters,
    prepare_detection,
    validate_yolo,
)


def make_detection(tmp_path, count=12):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(count):
        cv2.imwrite(str(images / f"{i}.png"), np.full((20, 40, 3), i, np.uint8))
        (labels / f"{i}.txt").write_text("0 0.5 0.5 0.6 0.4\n")
    return images, labels


def test_detection_split_hash_integrity_and_determinism(tmp_path):
    images, labels = make_detection(tmp_path)
    first = prepare_detection(images, labels, tmp_path / "a")
    second = prepare_detection(images, labels, tmp_path / "b")
    assert first.read_text() == second.read_text()
    records = load_manifest(first)
    assert {r["split"] for r in records} == {"train", "val", "test"}
    assert len(records) == 12
    changed = first.parent / records[0]["label_path"]
    changed.write_text("")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(first)


@pytest.mark.parametrize(
    "line",
    ["1 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .1 .5 .8 .2", "0 .5 .5 0 .2", "0 .5 .5 .2", "a b c d e"],
)
def test_bad_yolo_annotations(tmp_path, line):
    path = tmp_path / "label.txt"
    path.write_text(line)
    with pytest.raises(ValueError):
        validate_yolo(path)


def test_missing_labels_do_not_become_negatives(tmp_path):
    images, labels = make_detection(tmp_path)
    (labels / "0.txt").unlink()
    with pytest.raises(ValueError, match="Missing label"):
        prepare_detection(images, labels, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_leakage_rejected_even_when_reading_train(tmp_path):
    images, labels = make_detection(tmp_path)
    manifest = prepare_detection(images, labels, tmp_path / "out")
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    rows[1]["group"] = rows[0]["group"]
    rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(manifest, "train")


def test_characters_explicit_vocabulary_and_stratification(tmp_path):
    source = tmp_path / "chars"
    for index, c in enumerate(VOCABULARY):
        folder = source / c
        folder.mkdir(parents=True)
        for j in range(3):
            image = np.zeros((28, 28, 3), np.uint8)
            image[3:25, 5:20] = index * 3 + j + 1
            cv2.imwrite(str(folder / f"{j}.png"), image)
    manifest = prepare_characters(source, tmp_path / "prepared")
    for split in ("train", "val", "test"):
        assert {r["label"] for r in load_manifest(manifest, split)} == set(VOCABULARY)
    assert json.loads((manifest.parent / "dataset.json").read_text())["vocabulary"] == VOCABULARY


def test_numeric_mapping_is_never_guessed(tmp_path):
    with pytest.raises(ValueError, match="Missing character folder"):
        prepare_characters(tmp_path, tmp_path / "out")


def test_conflicting_glyph_quarantine_excludes_every_copy_and_preserves_raw(tmp_path):
    source = tmp_path / "raw"
    for index, character in enumerate(VOCABULARY):
        directory = source / character
        directory.mkdir(parents=True)
        for i in range(4):
            canvas = np.zeros((28, 28, 3), np.uint8)
            canvas[2:25, 4:20] = index * 4 + i + 1
            cv2.imwrite(str(directory / f"{i}.png"), canvas)
    duplicate = (source / "0/0.png").read_bytes()
    (source / "O/conflict.png").write_bytes(duplicate)
    with pytest.raises(ValueError, match="different character labels"):
        prepare_characters(source, tmp_path / "strict")
    manifest = prepare_characters(source, tmp_path / "clean", quarantine_conflicts=True)
    report = json.loads((manifest.parent / "quarantine.json").read_text())
    assert report["excluded_samples"] == 2 and report["conflicting_hashes"] == 1
    records = load_manifest(manifest)
    assert len(records) == 143
    assert (source / "0/0.png").read_bytes() == duplicate
    assert (source / "O/conflict.png").is_file()
    assert all(r["source"] not in {"0/0.png", "O/conflict.png"} for r in records)
