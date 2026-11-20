import cv2
import numpy as np
import pytest

from alpr.segmentation import SegmentationConfig, center_glyph, foreground, segment


def plate(text="AB08", inverse=False):
    image = np.full((75, 333, 3), 255, np.uint8)
    for i, character in enumerate(text):
        cv2.putText(
            image,
            character,
            (30 + i * 60, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            3,
            cv2.LINE_8,
        )
    return 255 - image if inverse else image


def test_contour_holes_do_not_duplicate_characters():
    glyphs, debug = segment(plate())
    assert len(glyphs) == 4  # B, 0, 8 each contain holes.
    assert [g.box[0] for g in glyphs] == sorted(g.box[0] for g in glyphs)
    assert debug["mask"].shape == (75, 333)
    assert all(g.image.shape == (44, 44) for g in glyphs)


def test_polarity_and_empty_plate():
    normal, _ = segment(plate(), SegmentationConfig(polarity="auto"))
    inverted, _ = segment(plate(inverse=True), SegmentationConfig(polarity="auto"))
    assert len(normal) == len(inverted) == 4
    assert not segment(np.full((75, 333, 3), 255, np.uint8))[0]


def test_two_rows_are_read_in_order():
    image = np.full((150, 333, 3), 255, np.uint8)
    for x, y in [(100, 60), (160, 60), (30, 135), (90, 135)]:
        cv2.putText(image, "A", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    glyphs, _ = segment(image, SegmentationConfig(height=150))
    assert len(glyphs) == 4
    assert glyphs[0].box[0] > glyphs[2].box[0]
    assert glyphs[0].box[1] < glyphs[2].box[1]


def test_center_glyph_retains_aspect_and_padding():
    image = np.zeros((80, 40), np.uint8)
    image[5:75, 18:22] = 255
    result = center_glyph(image)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(result))
    assert h == 40 and w < 5 and y >= 2 and x >= 2
    assert np.array_equal(result, center_glyph(foreground(result)))


@pytest.mark.parametrize(
    "kwargs",
    [{"kernel_size": 2}, {"border": 40}, {"min_height": 1}, {"method": "bogus"}, {"threshold": -1}],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SegmentationConfig(**kwargs)
