"""
Causal streaming LSTM for temporal modeling and next-feature prediction.

Implements a unidirectional LSTM for stateful streaming inference.
Predicts the next feature representation for anomaly detection.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalLSTM(nn.Module):
    """
    Causal streaming LSTM for temporal modeling.

    - 2-layer unidirectional LSTM
    - Input size: 512 (fused feature dimension)
    - Hidden size: 256
    - Dropout: 0.2 between layers
    - Supports stateful streaming inference with hidden state persistence
    """

    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize causal LSTM.

        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout between layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Unidirectional for causal behavior
        )

        logger.info(f"Causal LSTM: input={input_size}, hidden={hidden_size}, layers={num_layers}")

    def forward(self,
                x: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor (B, T, input_size) or (B, input_size) for single step
            hidden_state: Previous hidden state (h, c) or None

        Returns:
            Tuple of (output, (h, c)) where output is (B, T, hidden_size) or (B, hidden_size)
        """
        # Track if input was originally 2-D
        was_2d = x.dim() == 2

        # Ensure input has 3 dimensions: (batch, seq, input_size)
        if was_2d:
            x = x.unsqueeze(1)  # Add sequence dimension

        if hidden_state is None:
            # Initialize hidden state to zeros
            batch_size = x.shape[0]
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden_state = (h0, c0)

        output, (h, c) = self.lstm(x, hidden_state)

        # If input was originally 2-D, squeeze output back to 2-D
        if was_2d:
            output = output.squeeze(1)

        return output, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Batch size
            device: Device for tensors

        Returns:
            Tuple of (h, c) initialized to zeros
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


class NextFeaturePredictor(nn.Module):
    """
    Prediction head for next-feature prediction.

    Takes LSTM hidden state and predicts the next feature representation.
    """

    def __init__(self,
                 hidden_size: int = 256,
                 output_size: int = 512):
        """
        Initialize prediction head.

        Args:
            hidden_size: LSTM hidden size
            output_size: Output feature dimension (same as input to LSTM)
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

        logger.info(f"Next-feature predictor: hidden={hidden_size}, output={output_size}")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict the next feature from an LSTM hidden representation.

        The input may be a single hidden vector ``(B, hidden_size)`` or a
        sequence of LSTM outputs ``(B, T, hidden_size)``.  The predictor is a
        point-wise MLP, so it is applied to the last dimension in both cases.
        ``TemporalModel`` is responsible for selecting the last LSTM layer
        before calling this module.

        Args:
            hidden_state: Hidden representation with shape ``(..., hidden_size)``

        Returns:
            Predicted next feature with shape ``(..., output_size)``
        """
        return self.predictor(hidden_state)


class TemporalModel(nn.Module):
    """
    Complete temporal model combining LSTM and prediction head.

    Learns normal temporal dynamics and predicts next features.
    """

    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize temporal model.

        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout between layers
        """
        super().__init__()

        self.lstm = CausalLSTM(input_size, hidden_size, num_layers, dropout)
        self.predictor = NextFeaturePredictor(hidden_size, input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

        logger.info("Temporal model initialized (LSTM + prediction head)")

    def forward(self,
                x: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: LSTM -> prediction.

        Args:
            x: Input tensor (B, T, input_size) or (B, input_size)
            hidden_state: Previous hidden state or None

        Returns:
            Tuple of (prediction, (h, c))
        """
        output, (h, c) = self.lstm(x, hidden_state)
        # ``h`` has shape (num_layers, B, hidden_size); use the last layer.
        prediction = self.predictor(h[-1])
        return prediction, (h, c)

    def forward_sequence(self,
                         x: torch.Tensor,
                         hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                         ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict one next-feature vector for every timestep in a sequence.

        This is the training/evaluation path.  It is causal: the prediction
        at position ``t`` depends only on inputs up to and including ``t``.

        Args:
            x: Input sequence (B, T, input_size)
            hidden_state: Optional previous hidden state

        Returns:
            predictions: (B, T, input_size)
            hidden_state: final (h, c)
        """
        if x.dim() != 3:
            raise ValueError(f"forward_sequence expects (B, T, D); got {tuple(x.shape)}")
        output, (h, c) = self.lstm(x, hidden_state)
        predictions = self.predictor(output)
        return predictions, (h, c)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        return self.lstm.init_hidden(batch_size, device)
