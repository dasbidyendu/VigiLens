"""Tests for the causal LSTM and next-feature predictor."""

import torch

from models.temporal_lstm import CausalLSTM, NextFeaturePredictor, TemporalModel


def test_next_feature_predictor_accepts_vectors_and_sequences():
    predictor = NextFeaturePredictor(hidden_size=32, output_size=16)
    vectors = torch.randn(4, 32)
    sequences = torch.randn(4, 7, 32)

    assert predictor(vectors).shape == (4, 16)
    assert predictor(sequences).shape == (4, 7, 16)


def test_causal_lstm_hidden_shapes():
    lstm = CausalLSTM(input_size=16, hidden_size=8, num_layers=2, dropout=0.0)
    sequence = torch.randn(3, 5, 16)
    output, (h, c) = lstm(sequence)

    assert output.shape == (3, 5, 8)
    assert h.shape == (2, 3, 8)
    assert c.shape == (2, 3, 8)


def test_temporal_model_forward_sequence_shapes():
    model = TemporalModel(input_size=16, hidden_size=8, num_layers=2, dropout=0.0)
    sequence = torch.randn(2, 6, 16)
    predictions, (h, c) = model.forward_sequence(sequence)

    assert predictions.shape == (2, 6, 16)
    assert h.shape == (2, 2, 8)
    assert c.shape == (2, 2, 8)


def test_temporal_model_is_causal():
    """Predictions at time t must not depend on inputs after t."""
    torch.manual_seed(0)
    model = TemporalModel(input_size=8, hidden_size=4, num_layers=1, dropout=0.0)
    model.eval()
    sequence = torch.randn(1, 10, 8)

    full, _ = model.forward_sequence(sequence)
    prefix, _ = model.forward_sequence(sequence[:, :5])
    changed_future = sequence.clone()
    changed_future[:, 5:] = torch.randn_like(changed_future[:, 5:])
    full_changed, _ = model.forward_sequence(changed_future)

    assert torch.allclose(full[:, :5], prefix, atol=1e-6)
    assert torch.allclose(full[:, :5], full_changed[:, :5], atol=1e-6)
    assert not torch.allclose(full[:, 5:], full_changed[:, 5:], atol=1e-6)


def test_temporal_model_single_step_forward():
    model = TemporalModel(input_size=8, hidden_size=4, num_layers=1, dropout=0.0)
    step = torch.randn(2, 8)
    prediction, (h, c) = model(step)
    assert prediction.shape == (2, 8)
    assert h.shape == (1, 2, 4)
    assert c.shape == (1, 2, 4)
