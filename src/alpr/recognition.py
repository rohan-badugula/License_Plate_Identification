"""Character networks with checkpoints that carry their decoding contract."""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from .data import VOCABULARY
from .segmentation import center_glyph, foreground

ARCHITECTURES = ("efficientnet_b7", "efficientnet_b0", "mobilenet_v2", "resnet50", "inception_v3")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RecognitionConfig:
    architecture: str = "efficientnet_b7"
    image_size: int = 80
    vocabulary: str = VOCABULARY
    glyph_threshold: int = 200
    glyph_method: str = "otsu"
    glyph_polarity: str = "auto"
    mean: tuple = MEAN
    std: tuple = STD

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if self.image_size < 32 or (self.architecture == "inception_v3" and self.image_size < 75):
            raise ValueError("Image size must be >=32 (>=75 for InceptionV3)")
        if self.vocabulary != VOCABULARY:
            raise ValueError("Checkpoint must use the explicit 36-character vocabulary")
        if self.glyph_method not in {"fixed", "otsu"}:
            raise ValueError("Invalid glyph threshold method")
        if self.glyph_polarity not in {"auto", "dark", "light"}:
            raise ValueError("Invalid glyph polarity")
        if not 0 <= self.glyph_threshold <= 255:
            raise ValueError("Invalid glyph threshold")
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise ValueError(
                "Expected three normalization channels and positive standard deviations"
            )


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


class CharacterNetwork(nn.Module):
    def __init__(self, config: RecognitionConfig, pretrained: bool = False):
        super().__init__()
        constructor = getattr(models, config.architecture)
        weights = models.get_model_weights(constructor).DEFAULT if pretrained else None
        kwargs = {"weights": weights}
        if config.architecture == "inception_v3":
            kwargs.update(init_weights=False, aux_logits=bool(pretrained), transform_input=False)
        self.backbone = constructor(**kwargs)
        if config.architecture in {"resnet50", "inception_v3"}:
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            if config.architecture == "inception_v3":
                self.backbone.aux_logits = False
                self.backbone.AuxLogits = None
                self.backbone.transform_input = False
        else:
            features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(config.vocabulary)),
        )

    def forward(self, image):
        return self.head(self.backbone(image))


def glyph_tensor(
    image: np.ndarray, config: RecognitionConfig, segmented: bool = False
) -> torch.Tensor:
    # Segmentation already produces a foreground mask; raw training glyphs use explicit polarity.
    mask = (
        image
        if segmented
        else foreground(image, config.glyph_threshold, config.glyph_method, config.glyph_polarity)
    )
    mask = center_glyph(mask)
    resized = cv2.resize(
        mask, (config.image_size, config.image_size), interpolation=cv2.INTER_LINEAR
    )
    rgb = np.repeat(resized[None, :, :], 3, axis=0).astype(np.float32) / 255.0
    rgb = (rgb - np.array(config.mean, dtype=np.float32)[:, None, None]) / np.array(
        config.std, dtype=np.float32
    )[:, None, None]
    return torch.from_numpy(rgb.copy())


def save_checkpoint(
    path: Path, model: CharacterNetwork, config: RecognitionConfig, **metadata
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "config": asdict(config),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            **metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[CharacterNetwork, RecognitionConfig, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Recognition checkpoint not found: {path}; train a recognizer first"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported recognition checkpoint format; expected paper-alpr format v1")
    config = RecognitionConfig(**checkpoint["config"])
    model = CharacterNetwork(config, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, config, metadata


class CharacterRecognizer:
    def __init__(self, checkpoint: Path, device: str = "auto", batch_size: int = 32):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.device = resolve_device(device)
        self.model, self.config, self.metadata = load_checkpoint(checkpoint, self.device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict(self, glyphs: list[np.ndarray]) -> list[dict]:
        output = []
        for start in range(0, len(glyphs), self.batch_size):
            batch = torch.stack(
                [
                    glyph_tensor(g, self.config, segmented=True)
                    for g in glyphs[start : start + self.batch_size]
                ]
            ).to(self.device)
            probabilities = self.model(batch).softmax(dim=1)
            confidence, indices = probabilities.max(dim=1)
            for index, score in zip(indices.tolist(), confidence.tolist(), strict=True):
                output.append({"character": self.config.vocabulary[index], "confidence": score})
        return output
