"""Evaluation that includes misses and false positives, with explicit denominators."""

import math

import numpy as np


def classification_metrics(truth: list[int], predicted: list[int], classes: str) -> dict:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Expected equally sized, nonempty classification lists")
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        if not 0 <= actual < n or not 0 <= guess < n:
            raise ValueError("Class index out of range")
        matrix[actual, guess] += 1
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(n), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) != 0
    )
    result = {
        "samples": len(truth),
        "accuracy": float(tp.sum() / len(truth)),
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        "zero_division": 0,
        "macro_includes_all_classes": True,
    }
    for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result[f"macro_{name}"] = float(values.mean())
        result[f"weighted_{name}"] = float(np.average(values, weights=support))
    result["per_class"] = {
        c: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }
    return result


def edit_distance(actual: str, predicted: str) -> int:
    previous = list(range(len(predicted) + 1))
    for i, a in enumerate(actual, 1):
        current = [i]
        for j, b in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def box_iou(a: list | tuple, b: list | tuple) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def normalize_text(text: str) -> str:
    return "".join(c for c in text.upper() if c not in " -\t\r\n")


def plate_metrics(samples: list[dict], iou_threshold: float = 0.5) -> dict:
    """Greedy confidence-ordered one-to-one IoU matching (no use of text in matching)."""
    if not samples or not 0 < iou_threshold <= 1:
        raise ValueError("Need samples and an IoU threshold in (0, 1]")
    total = matched = exact = false_positive = missed = errors = characters = image_exact = 0
    for sample in samples:
        truth, predictions = sample["truth"], sample["predictions"]
        remaining = set(range(len(truth)))
        total += len(truth)
        correct_in_image = 0
        for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
            candidate = max(
                remaining, key=lambda i: box_iou(truth[i]["box"], prediction["box"]), default=None
            )
            if (
                candidate is None
                or box_iou(truth[candidate]["box"], prediction["box"]) < iou_threshold
            ):
                false_positive += 1
                errors += len(normalize_text(prediction["text"]))
                continue
            remaining.remove(candidate)
            matched += 1
            actual, guess = (
                normalize_text(truth[candidate]["text"]),
                normalize_text(prediction["text"]),
            )
            characters += len(actual)
            errors += edit_distance(actual, guess)
            if actual == guess:
                exact += 1
                correct_in_image += 1
        for index in remaining:
            missed += 1
            length = len(normalize_text(truth[index]["text"]))
            characters += length
            errors += length
        if correct_in_image == len(truth) and len(predictions) == len(truth):
            image_exact += 1
    precision = matched / (matched + false_positive) if matched + false_positive else 0.0
    return {
        "images": len(samples),
        "ground_truth_plates": total,
        "matched_plates": matched,
        "missed_plates": missed,
        "false_positive_plates": false_positive,
        "exact_plate_matches": exact,
        "plate_exact_match_rate": exact / total if total else None,
        "image_exact_match_rate": image_exact / len(samples),
        "detection_precision": precision,
        "detection_recall": matched / total if total else None,
        "character_edits": errors,
        "ground_truth_characters": characters,
        "character_error_rate": errors / characters if characters else None,
        "iou_threshold": iou_threshold,
        "matching": "confidence-ordered greedy one-to-one IoU",
        "normalization": "uppercase; remove ASCII spaces, hyphens and whitespace",
    }


def validate_box(box: list) -> None:
    if (
        len(box) != 4
        or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box)
        or box[0] < 0
        or box[1] < 0
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"Expected finite positive-area xyxy box: {box}")
