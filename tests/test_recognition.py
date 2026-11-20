import numpy as np
import pytest
import torch

from alpr.recognition import (
    ARCHITECTURES,
    CharacterNetwork,
    CharacterRecognizer,
    RecognitionConfig,
    glyph_tensor,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_training_and_segmented_preprocessing_are_identical():
    image = np.zeros((44, 44), np.uint8)
    image[4:39, 16:26] = 255
    config = RecognitionConfig()
    assert torch.equal(glyph_tensor(image, config), glyph_tensor(image, config, segmented=True))
    assert glyph_tensor(image, config).shape == (3, 80, 80)


@pytest.mark.integration
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_model_forward_without_downloads(architecture):
    config = RecognitionConfig(architecture=architecture)
    network = CharacterNetwork(config).eval()
    with torch.inference_mode():
        logits = network(torch.zeros(2, 3, config.image_size, config.image_size))
    assert logits.shape == (2, 36)
    assert torch.isfinite(logits).all()


@pytest.mark.integration
def test_checkpoint_roundtrip_and_glyph_batching(tmp_path):
    config = RecognitionConfig(architecture="efficientnet_b0")
    model = CharacterNetwork(config).eval()
    path = tmp_path / "recognizer.pt"
    save_checkpoint(path, model, config, training_status="random smoke fixture")
    restored, decoded, metadata = load_checkpoint(path, torch.device("cpu"))
    assert decoded == config
    assert metadata["training_status"] == "random smoke fixture"
    tensor = torch.randn(1, 3, 80, 80)
    with torch.inference_mode():
        assert torch.equal(model(tensor), restored(tensor))
    recognizer = CharacterRecognizer(path, "cpu", batch_size=2)
    assert recognizer.predict([]) == []
    predictions = recognizer.predict([np.zeros((44, 44), np.uint8)] * 3)
    assert len(predictions) == 3
    assert all(0 <= p["confidence"] <= 1 for p in predictions)


def test_bad_config_and_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError):
        RecognitionConfig(vocabulary="ABC")
    with pytest.raises(FileNotFoundError, match="train a recognizer"):
        load_checkpoint(tmp_path / "absent.pt", torch.device("cpu"))
