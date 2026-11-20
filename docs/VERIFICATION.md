# Verification ledger

Local verification date: November 20, 2026.

## Software and hardware

- Python 3.13.3, macOS arm64, Apple M3 Pro.
- PyTorch 2.15.0, torchvision 0.29.2, Ultralytics 8.4.216,
  OpenCV 4.14.0.94, NumPy 2.5.5.
- CPU paths tested. Apple MPS is available when the process can access the GPU;
  the restricted sandbox reported it unavailable. CUDA is unavailable on this Mac.
- `requirements-verified.txt` captures the installed package snapshot. It is a
  platform-specific record, not a guarantee of portable CUDA-wheel resolution.

## Completed checks

- 48 pytest tests passed locally, including all five real classifier architectures,
  checkpoint save/load and batching, archived YOLO loading/prediction, image/video
  paths, dataset integrity, contour behavior and metric denominators.
- Ruff lint and formatting checks passed.
- `pip check` found no dependency conflicts.
- Both training commands ran on synthetic fixtures without pretrained downloads.
- B7 full-backbone training and frozen-backbone training each completed a one-epoch
  CPU smoke run with augmentation. Checkpoints were reloaded for held-out evaluation.
- Recognition evaluation used batch size 7 on 36 samples, exercising the final
  partial batch. The report includes all 36.
- Fresh YOLOv8n trained for one epoch on a tiny synthetic detection fixture and
  evaluated on its separate test partition.
- `scripts/smoke_test.py` completed all data-preparation, training, component-test,
  image-inference and full-image-evaluation steps using synthetic data.
- Versioned archives for Mercosur, characters and OpenALPR were downloaded,
  extracted and hashed; source-specific audits are in `DATA_AUDIT.json`.
- Full public datasets were prepared; all 36 classes remain represented in all
  character splits after documented conflict quarantine.

Synthetic model accuracy is intentionally not reported as a research result.
No passing software test demonstrates the paper's 99.5%, 98.22% or 96.6% figures.

## Full real-data experiments

Full B7 training has been started on the Mac GPU using all 30,233 prepared training
glyphs, 15 epochs, batch size 64, ImageNet initialization, Adam at 1e-4, 80×80 inputs,
and validation-only checkpoint selection. Training and subsequent evaluation status
will be recorded here when verified. This initial entry does not assert completion.

Fresh full-data detector training, held-out component metrics and end-to-end benchmark
results must be labeled with their actual run status. Historical detector metrics are
not substituted for a new run. The paper's private benchmark remains unavailable.

## Hosted CI

The workflow covers Python 3.11/3.13 on Ubuntu, with lint, formatting, tests and package
builds. A committed workflow is not evidence that a hosted run passed; inspect the
actual GitHub check status on the pull request.
