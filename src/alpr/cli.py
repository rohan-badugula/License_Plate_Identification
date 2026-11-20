"""One CLI for dataset preparation, training, evaluation and image/video inference."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .data import prepare_characters, prepare_detection
from .io import environment, sha256, write_json
from .segmentation import SegmentationConfig


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="alpr", description=__doc__)
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show dependencies and compute devices")
    for kind in ("detection", "characters"):
        p = sub.add_parser(f"prepare-{kind}", help=f"Validate and split {kind} data")
        if kind == "detection":
            p.add_argument("--images", type=Path, required=True)
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--groups", type=Path, help="CSV path,group for video/vehicle grouping")
        else:
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--mapping", type=Path, help="JSON mapping folder name to character")
            p.add_argument(
                "--quarantine-conflicts",
                action="store_true",
                help="Exclude all copies of byte-identical glyphs with conflicting labels",
            )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--val-fraction", type=float, default=0.1)
        p.add_argument("--test-fraction", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("train-detector", help="Fine-tune YOLOv8 with validated data")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p = sub.add_parser("train-recognizer", help="Train the 36-class character recognizer")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--architecture",
        choices=["efficientnet_b7", "efficientnet_b0", "mobilenet_v2", "resnet50", "inception_v3"],
        default="efficientnet_b7",
    )
    p.add_argument("--image-size", type=int, default=80)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--glyph-polarity", choices=["auto", "dark", "light"], default="auto")
    p.add_argument("--glyph-method", choices=["fixed", "otsu"], default="otsu")
    p.add_argument("--glyph-threshold", type=int, default=200)
    for kind in ("detector", "recognizer"):
        p = sub.add_parser(f"evaluate-{kind}")
        p.add_argument("--data" if kind == "detector" else "--manifest", type=Path, required=True)
        p.add_argument("--weights", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--split", choices=["val", "test"], default="test")
        p.add_argument("--device", default="cpu")
        if kind == "detector":
            p.add_argument("--image-size", type=int, default=640)
        else:
            p.add_argument("--batch-size", type=int, default=32)
    for name in ("infer", "video", "evaluate"):
        p = sub.add_parser(name, help="Full YOLOv8 → segmentation → EfficientNet pipeline")
        p.add_argument("--detector", type=Path, required=True)
        p.add_argument("--recognizer", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--confidence", type=float, default=0.25)
        p.add_argument("--nms-iou", type=float, default=0.7)
        p.add_argument("--image-size", type=int, default=640)
        p.add_argument("--min-character-confidence", type=float, default=0.0)
        p.add_argument("--segmentation", type=Path, help="JSON object of SegmentationConfig fields")
        if name == "evaluate":
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--match-iou", type=float, default=0.5)
        else:
            p.add_argument("--source", type=Path, required=True)
        if name == "infer":
            p.add_argument("--debug", action="store_true")
        if name == "video":
            p.add_argument("--stride", type=int, default=1)
    return root


def execute(a: argparse.Namespace):
    if a.command == "doctor":
        import torch

        return {
            **environment(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
    if a.command.startswith("prepare-"):
        options = {"val": a.val_fraction, "test": a.test_fraction, "seed": a.seed}
        manifest = (
            prepare_detection(a.images, a.labels, a.output, groups_file=a.groups, **options)
            if a.command == "prepare-detection"
            else prepare_characters(
                a.source,
                a.output,
                mapping_file=a.mapping,
                quarantine_conflicts=a.quarantine_conflicts,
                **options,
            )
        )
        return {"manifest": str(manifest)}
    if a.command == "train-recognizer":
        from .recognition import RecognitionConfig
        from .training import train_recognizer

        config = RecognitionConfig(
            architecture=a.architecture,
            image_size=a.image_size,
            glyph_polarity=a.glyph_polarity,
            glyph_method=a.glyph_method,
            glyph_threshold=a.glyph_threshold,
        )
        return train_recognizer(
            a.manifest,
            a.output,
            config,
            epochs=a.epochs,
            batch_size=a.batch_size,
            learning_rate=a.learning_rate,
            seed=a.seed,
            device=a.device,
            pretrained=not a.no_pretrained,
            freeze_backbone=a.freeze_backbone,
            augment=not a.no_augment,
        )
    if a.command == "train-detector":
        from .detection import train_detector

        return train_detector(
            a.data,
            a.output,
            model=a.model,
            epochs=a.epochs,
            batch_size=a.batch_size,
            image_size=a.image_size,
            device=a.device,
            seed=a.seed,
            pretrained=not a.no_pretrained,
        )
    if a.command == "evaluate-recognizer":
        from .training import evaluate_recognizer

        metrics = evaluate_recognizer(
            a.manifest, a.weights, a.output, split=a.split, batch_size=a.batch_size, device=a.device
        )
        return {k: v for k, v in metrics.items() if k not in {"confusion_matrix", "per_class"}}
    if a.command == "evaluate-detector":
        from .detection import evaluate_detector

        return evaluate_detector(
            a.data, a.weights, a.output, split=a.split, image_size=a.image_size, device=a.device
        )
    from .detection import PlateDetector
    from .pipeline import ALPRPipeline, evaluate_pipeline, infer_images, infer_video
    from .recognition import CharacterRecognizer, resolve_device

    segmentation = (
        SegmentationConfig(**json.loads(a.segmentation.read_text())) if a.segmentation else None
    )
    recognizer = CharacterRecognizer(a.recognizer, a.device)
    detector = PlateDetector(
        a.detector,
        confidence=a.confidence,
        iou=a.nms_iou,
        image_size=a.image_size,
        device=str(resolve_device(a.device)),
    )
    pipeline = ALPRPipeline(detector, recognizer, segmentation, a.min_character_confidence)
    if a.command == "infer":
        result = infer_images(pipeline, a.source, a.output, debug=a.debug)
    elif a.command == "video":
        result = infer_video(pipeline, a.source, a.output, stride=a.stride)
    else:
        result = evaluate_pipeline(pipeline, a.manifest, a.output, iou_threshold=a.match_iou)
    # Persist weight identities for every full-pipeline operation.
    write_json(
        a.output / "weights.json",
        {
            "detector": str(a.detector),
            "detector_sha256": sha256(a.detector),
            "recognizer": str(a.recognizer),
            "recognizer_sha256": sha256(a.recognizer),
        },
    )
    return (
        {"output": str(a.output), "images": len(result["images"])}
        if a.command == "infer"
        else {k: v for k, v in result.items() if k != "samples"}
    )


def main() -> int:
    arguments = parser().parse_args()
    try:
        result = execute(arguments)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"alpr: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
