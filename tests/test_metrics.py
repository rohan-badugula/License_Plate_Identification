import pytest

from alpr.metrics import classification_metrics, edit_distance, plate_metrics


def test_multiclass_metrics_include_false_positives_and_missing_classes():
    metrics = classification_metrics([0, 0, 1], [0, 1, 1], "ABC")
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["B"]["precision"] == 0.5
    assert metrics["per_class"]["C"]["support"] == 0
    assert metrics["macro_recall"] == 0.5


def test_plate_evaluation_misses_extra_predictions_and_transcription():
    a = {"box": [0, 0, 100, 30], "text": "AB12"}
    b = {"box": [0, 50, 100, 80], "text": "XY34"}
    sample = {
        "truth": [a, b],
        "predictions": [
            {**a, "text": "ab-12", "confidence": 0.9},
            {"box": [200, 0, 300, 30], "text": "BAD", "confidence": 0.8},
        ],
    }
    metrics = plate_metrics([sample])
    assert metrics["plate_exact_match_rate"] == 0.5
    assert metrics["image_exact_match_rate"] == 0
    assert metrics["missed_plates"] == metrics["false_positive_plates"] == 1
    assert metrics["character_error_rate"] == 7 / 8


def test_duplicate_detections_cannot_match_same_truth():
    target = {"box": [0, 0, 100, 30], "text": "A"}
    prediction = {**target, "confidence": 0.9}
    result = plate_metrics([{"truth": [target], "predictions": [prediction, prediction]}])
    assert result["matched_plates"] == 1
    assert result["detection_precision"] == 0.5
    assert result["image_exact_match_rate"] == 0


def test_negative_images_and_empty_cer_denominator():
    metrics = plate_metrics([{"truth": [], "predictions": []}])
    assert metrics["image_exact_match_rate"] == 1
    assert metrics["plate_exact_match_rate"] is None
    assert metrics["character_error_rate"] is None
    assert edit_distance("ABC123", "AC1234") == 2
