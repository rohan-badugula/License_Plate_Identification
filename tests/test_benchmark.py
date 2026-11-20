import json

import cv2
import numpy as np

from alpr.benchmark import prepare_openalpr


def test_benchmark_conversion_preserves_all_plates_and_documents_exclusions(tmp_path):
    source = tmp_path / "source/us"
    source.mkdir(parents=True)
    image = np.zeros((100, 200, 3), np.uint8)
    cv2.imwrite(str(source / "car.jpg"), image)
    (source / "car.txt").write_text(
        "car.jpg\t10\t20\t50\t30\tABC123\ncar.jpg\t100\t40\t40\t20\tXY12\n"
    )
    image[:] = 255
    cv2.imwrite(str(source / "bad.jpg"), image)
    (source / "bad.txt").write_text("bad.jpg\t0\t0\t20\t20\t汉字\n")
    output = tmp_path / "prepared"
    summary = prepare_openalpr(source.parent, output)
    assert summary["images"] == 1 and summary["plates"] == 2
    assert summary["by_region"] == {"us": 1}
    entry = json.loads((output / "benchmark.jsonl").read_text())
    assert entry["plates"][0]["box"] == [10, 20, 60, 50]
    assert len(entry["plates"]) == 2
    assert summary["excluded"][0]["reason"] == "unsupported or empty transcription"
