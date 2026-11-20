"""Contour segmentation with explicit polarity, filtering, and reading order."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    width: int = 333
    height: int = 75
    threshold: int = 200
    method: str = "fixed"
    polarity: str = "dark"
    kernel_size: int = 3
    border: int = 3
    min_height: float = 0.2
    max_height: float = 0.95
    min_width: float = 0.005
    max_width: float = 0.25
    min_aspect: float = 0.08
    max_aspect: float = 1.2
    row_tolerance: float = 0.6

    def __post_init__(self):
        if self.width < 8 or self.height < 8 or not 0 <= self.threshold <= 255:
            raise ValueError("Invalid segmentation image dimensions or threshold")
        if self.method not in {"fixed", "otsu"} or self.polarity not in {"dark", "light", "auto"}:
            raise ValueError("Invalid threshold method or polarity")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("Morphology kernel size must be positive and odd")
        if not 0 <= self.border < min(self.width, self.height) / 2:
            raise ValueError("Border must leave an interior region")
        if not (0 < self.min_height < self.max_height <= 1):
            raise ValueError("Invalid relative character height limits")
        if not (0 < self.min_width < self.max_width <= 1):
            raise ValueError("Invalid relative character width limits")
        if not (0 < self.min_aspect < self.max_aspect and self.row_tolerance > 0):
            raise ValueError("Invalid aspect ratio or row tolerance")


@dataclass
class Glyph:
    box: tuple[int, int, int, int]  # xyxy, resized plate coordinates
    image: np.ndarray  # white foreground on black background


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected grayscale or BGR image")


def foreground(
    image: np.ndarray, threshold: int = 200, method: str = "otsu", polarity: str = "auto"
) -> np.ndarray:
    gray = grayscale(image)
    flags = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if method == "otsu" else 0)
    _, mask = cv2.threshold(gray, threshold, 255, flags)
    if polarity == "auto":
        edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        polarity = "dark" if np.median(edge) > 127 else "light"
    return cv2.bitwise_not(mask) if polarity == "dark" else mask


def center_glyph(mask: np.ndarray, size: int = 44, padding: int = 2) -> np.ndarray:
    if size <= 2 * padding:
        raise ValueError("Glyph canvas must leave space for foreground")
    points = cv2.findNonZero(mask)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if points is None:
        return canvas
    x, y, width, height = cv2.boundingRect(points)
    cropped = mask[y : y + height, x : x + width]
    scale = (size - 2 * padding) / max(width, height)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)
    left, top = (size - width) // 2, (size - height) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


def reading_order(boxes: list[tuple], tolerance: float) -> list[tuple]:
    rows = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        center = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        row = next(
            (
                r
                for r in rows
                if abs(center - np.median([(b[1] + b[3]) / 2 for b in r]))
                < tolerance * min(height, np.median([b[3] - b[1] for b in r]))
            ),
            None,
        )
        if row is None:
            rows.append([box])
        else:
            row.append(box)
    return [box for row in rows for box in sorted(row, key=lambda b: b[0])]


def segment(
    image: np.ndarray, config: SegmentationConfig | None = None
) -> tuple[list[Glyph], dict]:
    config = config or SegmentationConfig()
    gray = grayscale(image)
    resized = cv2.resize(gray, (config.width, config.height), interpolation=cv2.INTER_CUBIC)
    mask = foreground(resized, config.threshold, config.method, config.polarity)
    # Paper erodes then dilates the white plate background. Operate in that convention.
    background = cv2.bitwise_not(mask)
    kernel = np.ones((config.kernel_size, config.kernel_size), dtype=np.uint8)
    background = cv2.erode(background, kernel)
    background = cv2.dilate(background, kernel)
    mask = cv2.bitwise_not(background)
    border = config.border
    if border:
        mask[:border] = mask[-border:] = 0
        mask[:, :border] = mask[:, -border:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (
            config.min_height <= height / config.height <= config.max_height
            and config.min_width <= width / config.width <= config.max_width
            and config.min_aspect <= width / height <= config.max_aspect
        ):
            boxes.append((x, y, x + width, y + height))
    boxes = reading_order(boxes, config.row_tolerance)
    glyphs = [Glyph(b, center_glyph(mask[b[1] : b[3], b[0] : b[2]])) for b in boxes]
    return glyphs, {"resized": resized, "mask": mask}
