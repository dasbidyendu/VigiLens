"""Tests for modality-aware fusion."""

import torch

from models.fusion import ModalityAwareFusion


def test_fusion_shapes_and_modalities():
    torch.manual_seed(0)
    fusion = ModalityAwareFusion(input_dim=2048, hidden_dim=32, output_dim=64)
    rgb = torch.randn(4, 2048)
    ir = torch.randn(4, 2048)

    both = fusion(rgb, ir)
    rgb_only = fusion(rgb, None)
    ir_only = fusion(None, ir)

    assert both.shape == (4, 64)
    assert rgb_only.shape == (4, 64)
    assert ir_only.shape == (4, 64)
    assert torch.isfinite(both).all()
    assert torch.isfinite(rgb_only).all()
    assert torch.isfinite(ir_only).all()


def test_fusion_per_sample_availability():
    torch.manual_seed(0)
    fusion = ModalityAwareFusion(input_dim=16, hidden_dim=8, output_dim=4)
    rgb = torch.randn(3, 16)
    ir = torch.randn(3, 16)
    rgb_available = torch.tensor([True, True, False])
    ir_available = torch.tensor([True, False, True])

    fused = fusion(rgb, ir, rgb_available, ir_available)
    assert fused.shape == (3, 4)

    # Sample 0 uses both modalities; samples 1 and 2 use one each.
    assert torch.isfinite(fused).all()
    assert not torch.allclose(fused[0], fused[1])
    assert not torch.allclose(fused[1], fused[2])


def test_fusion_returns_zero_when_no_modality_available():
    fusion = ModalityAwareFusion(input_dim=16, hidden_dim=8, output_dim=4)
    rgb = torch.randn(2, 16)
    fused = fusion(rgb, None, torch.tensor([False, False]), False)
    assert torch.allclose(fused, torch.zeros_like(fused))


def test_fusion_requires_at_least_one_tensor():
    fusion = ModalityAwareFusion(input_dim=16, hidden_dim=8, output_dim=4)
    try:
        fusion(None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("Fusion should reject two missing modalities")


def test_fusion_gradients_flow():
    fusion = ModalityAwareFusion(input_dim=16, hidden_dim=8, output_dim=4)
    rgb = torch.randn(2, 16, requires_grad=True)
    ir = torch.randn(2, 16, requires_grad=True)
    loss = fusion(rgb, ir).sum()
    loss.backward()

    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert ir.grad is not None and ir.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in fusion.parameters()
    )
