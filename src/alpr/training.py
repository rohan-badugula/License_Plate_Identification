"""Train/validation isolation, full-batch evaluation, and recorded experiment provenance."""

import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomAffine

from .data import load_manifest
from .io import environment, new_run, read_image, sha256, write_json
from .metrics import classification_metrics
from .recognition import (
    CharacterNetwork,
    RecognitionConfig,
    glyph_tensor,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


class CharacterDataset(Dataset):
    def __init__(
        self, manifest: Path, split: str, config: RecognitionConfig, augment: bool = False
    ):
        self.records = load_manifest(manifest, split)
        self.root, self.config = manifest.parent, config
        if any(
            r.get("label") not in config.vocabulary or len(r.get("label", "")) != 1
            for r in self.records
        ):
            raise ValueError("Character manifest requires single-character vocabulary labels")
        self.augment = (
            RandomAffine(
                10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                interpolation=InterpolationMode.BILINEAR,
                fill=[-m / s for m, s in zip(config.mean, config.std, strict=True)],
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        tensor = glyph_tensor(read_image(self.root / record["path"]), self.config)
        if self.augment:
            tensor = self.augment(tensor)
        return tensor, self.config.vocabulary.index(record["label"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def evaluate_network(model, loader, config, device) -> dict:
    model.eval()
    loss = 0.0
    actual, predicted = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(actual, predicted, config.vocabulary)
    metrics["loss"] = loss / len(actual)
    return metrics


def train_recognizer(
    manifest: Path,
    output: Path,
    config: RecognitionConfig,
    *,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    augment: bool = True,
) -> dict:
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    seed_everything(seed)
    resolved = resolve_device(device)
    training = CharacterDataset(manifest, "train", config, augment)
    validation = CharacterDataset(manifest, "val", config)
    if {r["label"] for r in training.records} != set(config.vocabulary):
        raise ValueError("Training split must contain every character in the vocabulary")
    output = new_run(output)
    model = CharacterNetwork(config, pretrained=pretrained).to(resolved)
    if freeze_backbone:
        model.backbone.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    provenance = {
        "environment": environment(),
        "config": asdict(config),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(resolved),
        "pretrained": pretrained,
        "augment": augment,
        "freeze_backbone": freeze_backbone,
        "manifest_sha256": sha256(manifest),
        "manifest": str(manifest.resolve()),
        "selection_metric": "validation loss",
        "training_samples": len(training),
        "validation_samples": len(validation),
    }
    write_json(output / "experiment.json", provenance)
    best_loss, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone:
            model.backbone.eval()  # Do not update frozen BatchNorm statistics.
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(resolved), labels.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; inspect data and learning rate")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        metrics = evaluate_network(model, val_loader, config, resolved)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(training),
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} val_accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
        metadata = {
            "epoch": epoch,
            "validation": metrics,
            "provenance": provenance,
            "training_status": "trained",
        }
        save_checkpoint(output / "last.pt", model, config, **metadata)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(output / "best.pt", model, config, **metadata)
        write_json(output / "history.json", history)
    return {"checkpoint": str(output / "best.pt"), "epochs": epochs, "best_val_loss": best_loss}


def evaluate_recognizer(
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    device: str = "auto",
) -> dict:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    resolved = resolve_device(device)
    model, config, metadata = load_checkpoint(checkpoint, resolved)
    dataset = CharacterDataset(manifest, split, config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_network(model, loader, config, resolved)
    metrics.update(
        split=split,
        manifest_sha256=sha256(manifest),
        checkpoint_sha256=sha256(checkpoint),
        device=str(resolved),
        training_status=metadata.get("training_status", "unknown"),
        environment=environment(),
    )
    write_json(output, metrics)
    return metrics
