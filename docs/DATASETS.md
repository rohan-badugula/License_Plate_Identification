# Public data acquisition and provenance

Acquired and inspected on November 20, 2026. Raw archives and extracted images live
under ignored `data/downloads/`. The committed [registry](../configs/datasets.json)
pins versions and observed SHA-256 hashes. The [machine-readable audit](DATA_AUDIT.json)
records actual counts, split identities and exclusions without distributing images.

## Download and prepare

```bash
python scripts/download_datasets.py --dataset all
python scripts/prepare_public_data.py --dataset all --quarantine-conflicts
```

Preparation outputs `data/prepared/mercosur/`, `data/prepared/characters/`, and
`data/prepared/openalpr/`. Choose a new `--output` directory when rerunning preparation.
Downloads reuse existing verified archives. An interrupted extraction is not reused
silently; inspect and remove the partial extraction before retrying. Keep about 6 GB
available for archives, extracted data, prepared copies and a training run.

The explicit quarantine flag removes every copy of a byte-identical character image
assigned to different classes. Raw data remain unchanged. Each removed path, label,
and hash is recorded in `quarantine.json`. Without the flag preparation fails on
these conflicts. The source archive hash must match the registry; a changed upstream
archive requires investigation rather than silently updating the expected hash.

## Detection: Artificial Mercosur v2

- Publisher: [Mendeley Data](https://data.mendeley.com/datasets/nx9xbs4rgx/2).
- DOI: **10.17632/nx9xbs4rgx.2**; version 2, published July 15, 2020.
- Contributors: Gilles Velleneuve Trindade Silvano, Patricia Takako Endo,
  Vinícius Campos Tinoco Ribeiro, Vitor Rodrigues Greati, Ivanovitch Silva,
  Theo Lynn, and Aguinaldo Bezerra.
- Publisher license: **CC BY 4.0**; retain attribution and indicate changes.
- Downloaded archive: **1,634,711,438 bytes**.
- Actual image count: **3,832**; label files: **3,840**; one CSV.
- Eight labels have no corresponding image. They are listed in `source-audit.json`
  and are not treated as training samples. No missing image is fabricated.
- All available images decoded and their YOLO annotations passed validation.
- Prepared split: **3,066 train / 383 validation / 383 test**, seed 42.

The paper says 3,840 images; the linked dataset article describes 3,829. These
statements do not exactly match the downloaded v2 archive. Use measured counts for
this experiment and retain the original archive unchanged.

The publisher describes images acquired in Brazil from five acquisition categories:
monitoring cameras, several parking-lot capture sources, and cropped parking-lot
images. The five categories are not proof of images collected in five countries.
The Mercosur standard itself covers several countries. Resolutions vary; the archive
is not exclusively 800×600.

These splits group byte-identical files only. Neighboring monitoring frames, related
vehicles and cropped/uncropped scenes may remain correlated. Vehicle identities and
the original paper split are unavailable. Do not claim a vehicle-disjoint benchmark
from these manifests. Supply an independently reviewed `--groups` CSV when that
guarantee is required. The archived detector may have trained on images that appear
in these new partitions; only a freshly trained detector can use these partitions as
its own held-out evaluation.

## Characters: pinned upstream archive

- Source: [quangnhat185/Plate_detect_and_recognize](https://github.com/quangnhat185/Plate_detect_and_recognize).
- Pinned commit: `6943f30ebfc14ef27d982779e0c434f38114ada4`.
- Archive: `dataset_characters.zip`, **39,692,489 bytes**.
- Raw count: **37,623** images in 36 literal character folders.
- The count, class structure and sampled 28×28 images match the paper's description;
  some images have other dimensions. This is a strong candidate for the data family,
  not verified bit-for-bit recovery of the original Kaggle dataset version.
- Original notebook metadata includes Kaggle dataset/version identifiers, but the
  tested Kaggle public URLs/API endpoints did not provide a usable download here.
- The upstream repository includes an MIT license. Underlying image provenance is
  not fully described; preserve attribution and assess image rights separately
  before redistribution. This project downloads data locally and does not republish it.

The audit found **10 file hashes assigned to multiple character labels**, affecting
**30 source files**. Every copy of those hashes is quarantined; labels are not guessed.
The usable total is **37,593**. The prepared partitions contain **30,233 train /
3,639 validation / 3,721 test** images. Differences from exact 80/10/10 fractions
follow from assigning duplicate groups together within each class. The post-quarantine
class counts range from 1,005 to 1,085. Full exclusions are committed in `DATA_AUDIT.json`.

## Public full-image benchmark: OpenALPR

- Source: [openalpr/benchmarks](https://github.com/openalpr/benchmarks).
- Pinned commit: `9790ed20d475be1e940a7e2a3f492a97c2ebed9b`.
- Repository license: **AGPL-3.0**; preserve upstream notices.
- Archive: **217,824,648 bytes**.
- Converted from `endtoend/` tab-separated annotations: image, pixel x, y, width,
  height, and transcription. The new manifest uses pixel xyxy boxes.
- **443 usable images / 443 annotated plates:** 113 Brazilian, 108 European,
  and 222 US examples.
- `br/PJT2905.jpg` is excluded for an invalid ground-truth box. Twenty-three images
  without annotations (including `br/PAG5219.jpg` and the `usimages/` examples) are
  recorded and excluded rather than mislabeled as negatives.
- Every conversion decision, image hash and exclusion is saved in `conversion.json`.

This is **a different benchmark from the paper's 500 Indian phone images**. Its
country mix, layouts and fonts differ. Character-source provenance and relationships
between public datasets are incomplete, so independence from every upstream training
image cannot be guaranteed. Report benchmark identity, per-region results and these
limits; do not rename its result as a replication of 96.6%.

## What was not recovered

The original trained B7 `.h5` weights, exact historical split indices, and original
private 500-image benchmark were not found. ImageNet B7 initialization downloaded
for a new experiment is not the original trained ALPR recognizer. No substitute
weights, labels or benchmark scores are described as originals.
