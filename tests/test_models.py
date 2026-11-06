import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize(
    "mode,fusion",
    [("hybrid", "concat"), ("hybrid", "attention"), ("cnn", "none"), ("vit", "none")],
)
@pytest.mark.parametrize("nc", [2, 3])
def test_forward_shape(mode, fusion, nc):
    torch.manual_seed(42)
    model = build_model(mode, fusion, nc, pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, nc)


def test_gradcam_layer():
    assert build_model("hybrid", "attention", 2, pretrained=False).gradcam_layer is not None
    assert build_model("vit", "none", 2, pretrained=False).gradcam_layer is None
