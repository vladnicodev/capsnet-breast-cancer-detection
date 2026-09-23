import pytest
import torch

from capsnet_pcam.models import MODEL_NAMES, build_model, count_parameters


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 96, 96), torch.tensor([0, 1, 1, 0])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_backward(name, batch):
    x, y = batch
    model = build_model(name)
    out = model(x, y)
    assert out["score"].shape == (4,)
    assert torch.all((out["score"] >= 0) & (out["score"] <= 1))
    loss, parts = model.loss(out, x, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_eval_mode_needs_no_labels(name, batch):
    model = build_model(name).eval()
    with torch.no_grad():
        out = model(batch[0])
    assert out["score"].shape == (4,)


def test_capsnet_outputs(batch):
    x, y = batch
    model = build_model("capsnet").train()
    out = model(x, y)
    assert out["capsules"].shape == (4, 2, 16)
    assert out["recon"].shape == x.shape
    # The score is >= 0.5 exactly when the tumour capsule is the longer one.
    assert torch.equal(out["score"] >= 0.5, out["lengths"][:, 1] >= out["lengths"][:, 0])

    model.eval()
    assert "recon" not in model(x)
    assert model(x, reconstruct=True)["recon"].shape == x.shape


def test_capsnet_without_decoder(batch):
    model = build_model("capsnet", recon_weight=0)
    out = model(*batch)
    loss, parts = model.loss(out, *batch)
    assert "recon" not in out and set(parts) == {"margin"}


def test_matched_cnn_has_same_parameter_count_as_capsnet():
    capsnet = count_parameters(build_model("capsnet"))
    matched = count_parameters(build_model("matched_cnn"))
    # Only the biases/BatchNorm of the 9x9 layer and the final linear layer differ.
    assert abs(capsnet - matched) / capsnet < 0.001


def test_decoder_is_excluded_from_inference_parameters():
    model = build_model("capsnet")
    assert count_parameters(model, inference_only=False) > count_parameters(model)
    assert count_parameters(build_model("simple_cnn")) == 60_545  # same as the original Keras model
