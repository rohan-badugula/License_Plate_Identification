"""Convert OpenALPR's tab-delimited xywh annotations without inventing transcriptions."""

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .data import VOCABULARY
from .io import IMAGE_SUFFIXES, new_run, read_image, sha256, write_json
from .metrics import normalize_text, validate_box


def prepare_openalpr(source: Path, output: Path) -> dict:
    entries = defaultdict(list)
    excluded = []
    for annotation in sorted(source.rglob("*.txt")):
        with annotation.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        for row in rows:
            if len(row) != 6:
                raise ValueError(f"Expected image,x,y,width,height,text in {annotation}")
            name, x, y, width, height, text = row
            image = (annotation.parent / name).resolve()
            if image.parent != annotation.parent.resolve():
                raise ValueError("Benchmark image names must remain inside their region directory")
            x, y, width, height = map(int, (x, y, width, height))
            entries[image].append({"box": [x, y, x + width, y + height], "text": text})
    if not entries:
        raise ValueError("No OpenALPR benchmark annotations found")
    records, seen = [], {}
    for image, plates in entries.items():
        reason = None
        if not image.is_file():
            reason = "missing image"
        else:
            height, width = read_image(image).shape[:2]
            for plate in plates:
                try:
                    validate_box(plate["box"])
                except ValueError:
                    reason = "invalid ground-truth box"
                if plate["box"][2] > width or plate["box"][3] > height:
                    reason = "ground-truth box exceeds image"
                text = normalize_text(plate["text"])
                if not text or any(c not in VOCABULARY for c in text):
                    reason = "unsupported or empty transcription"
        if reason:
            excluded.append({"image": str(image.relative_to(source.resolve())), "reason": reason})
            continue
        digest = sha256(image)
        if digest in seen:
            # Conflicting copies are a data error; do not choose which truth to believe.
            if seen[digest] != plates:
                raise ValueError("Duplicate benchmark image has conflicting annotations")
            excluded.append(
                {
                    "image": str(image.relative_to(source.resolve())),
                    "reason": "duplicate image content with identical annotations",
                }
            )
            continue
        seen[digest] = plates
        records.append(
            {"source": image, "plates": plates, "region": image.parent.name, "sha256": digest}
        )
    if not records:
        raise ValueError("No usable benchmark images remain")
    output = new_run(output).resolve()
    result = []
    for record in records:
        image = record["source"]
        target = output / "images" / record["region"] / image.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        result.append(
            {
                "image": target.relative_to(output).as_posix(),
                "region": record["region"],
                "plates": record["plates"],
            }
        )
    manifest = output / "benchmark.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in result) + "\n", encoding="utf-8")
    all_images = {p.resolve() for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    summary = {
        "source": "OpenALPR public end-to-end benchmark; separate from the paper benchmark",
        "images": len(result),
        "plates": sum(len(r["plates"]) for r in result),
        "by_region": dict(Counter(r["region"] for r in result)),
        "excluded": excluded,
        "images_without_annotations": [
            str(p.relative_to(source.resolve())) for p in sorted(all_images - set(entries))
        ],
        "manifest_sha256": sha256(manifest),
        "source_hashes": {
            str(r["source"].relative_to(source.resolve())): r["sha256"] for r in records
        },
    }
    write_json(output / "conversion.json", summary)
    return summary
