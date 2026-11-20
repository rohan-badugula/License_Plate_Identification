# Paper and historical code audit

Source: *Encompassing YOLOv8 and EfficientNet B7 for Automatic License Plate
Identification*, ICCCCIS 2023, DOI [10.1109/ICCCIS60361.2023.10425775](https://doi.org/10.1109/ICCCIS60361.2023.10425775).
The original PDF, notebooks and experiment artifacts remain in their original locations.
This document distinguishes evidence from implementation decisions; it does not amend the paper.

| Topic | Evidence | Resolution in the runnable implementation |
| --- | --- | --- |
| Detection classes | Section IV-A and old README describe plate and background as two classes. | One foreground class, ID 0, `license_plate`. Background is implicit in YOLO; never annotate it as a second class. |
| Model variant | Paper does not specify n/s/m/l/x. Archived `Code/YOLO_V8_resluts/args.yaml` specifies `yolov8n.pt`. | Default YOLOv8n; variant is configurable and recorded. |
| Training epochs | Paper: 10 detection epochs, 15 recognition epochs. `character-recog.ipynb`: 25 recognition epochs plus early stopping. | Defaults 10 and 15, explicit overrides recorded. No assertion that old notebook reproduces paper exactly. |
| Splits | Paper: 80/20 train/test for each dataset. Character notebook: 90/10 train/validation. Detector notebook uses a held-out validation set. | Default 80/10/10 train/validation/test. Never use final test for checkpoint selection. This is an explicit methodological correction, not an exact historical replication. |
| Dataset identity | Paper cites Artificial Mercosur, 3,840 images, and 37,623 characters. Reference [26] is a paper, not a uniquely versioned dataset download. | Require local datasets; validate labels and persist content hashes and split membership. Do not invent dataset URLs or claim the original data were recovered. |
| Recognition architecture | Paper: ImageNet EfficientNet-B7, Adam, categorical cross-entropy, lr 1e-4. Notebook: 80x80 RGB, pooling, Dense(128), dropout 0.5, 36 outputs. | PyTorch EfficientNet-B7 with global average pooling, Dense(128), ReLU, dropout 0.5 and 36 logits; 80x80 default. Integer-label cross-entropy is mathematically equivalent to one-hot categorical loss. PyTorch weights and normalization differ from Keras; old `.h5` weights are not compatible. |
| Character labels | Paper only states numeric encoding 0-35; notebook uses fitted LabelEncoder. No class artifact is committed. | Explicit `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` vocabulary; a numeric folder dataset requires a user-supplied mapping JSON. Store vocabulary and preprocessing in every checkpoint. |
| Threshold | Paper specifies 200. Notebook combines threshold 200 with Otsu, which computes its own threshold. | Fixed 200 default for paper alignment; optional Otsu is identified as a different setting. |
| Morphology | Notebook passes `(3,3)` rather than an explicit structuring-element array; height/width variables are interchanged. | Explicit 3x3 kernel and named relative height/width filters. All morphology and contour filtering operate in memory. |
| Contours | Notebook retrieves tree contours, keeps top 15 by area, and sorts only by x; this may duplicate character holes and misorder two-line plates. | External contours on a foreground mask, configurable size filters, row clustering then left-to-right order. No hardcoded plate length or truncation. Two-line handling is an extension. |
| Inference integration | Character notebook uses Haar cascade plate crops, then loads a ResNet JSON and differently named weights after training B7. Detector notebook contains experimental EasyOCR paths. | A single YOLOv8 → segmentation → B7 pipeline; no OCR replacement or hidden fallback. Missing recognition weights fail clearly. |
| Classification metrics | Legacy rounded softmax metrics do not implement standard multiclass argmax macro precision/recall/F1. Averaging convention in the paper is not specified. | Argmax accuracy, per-class metrics, macro and weighted metrics, confusion matrix on all samples including partial batches. |
| Detection metric evidence | Archived final CSV row: mAP50 0.99496, mAP50-95 0.9795, precision 0.99411, recall 0.99454. Its args reference `facemask.yaml`, and dataset provenance is absent. Paper's quoted losses differ from this CSV. | Preserve artifacts as historical evidence, not fresh held-out evaluation. Independently evaluate any weights on a supplied test split. |
| End-to-end accuracy | Paper: 483/500 = 96.6%, Indian phone photos. Those labeled images and B7 weights are absent. | Full-image evaluation counts missed plates, extra plates, segmentation and transcription errors. No reproduced 96.6% claim without that data. |
| Speed and novelty | Old README claims lightweight real-time operation; paper claims first YOLOv8 LPR use. B7 has substantial compute requirements; no deployment latency evidence is supplied. | Do not repeat unverified novelty or real-time claims. Report local timing as measurements, with device details. |
| Licensing | Old README links a nonexistent MIT LICENSE. PDF is IEEE copyright; dependencies and datasets have separate terms. | No invented MIT grant. See `NOTICE.md`; preserve existing material and its attribution. |

## Reproduction boundary

This is an executable reconstruction and correction of the described method, not a
claim of bit-for-bit replication of a 2023 training environment. The original split
indices, dataset versions, B7 weights and 500-image benchmark are not in the repository.
Synthetic fixtures exercise software paths only and cannot establish recognition accuracy.

## References for implementation APIs

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics Python API](https://docs.ultralytics.com/usage/python/)
- [Torchvision EfficientNet-B7](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b7.html)
