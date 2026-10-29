"""
Modality-aware fusion module for RGB and IR features.

Supports RGB+IR, RGB-only, and IR-only modes with learnable fusion.
Availability can be a single boolean or a per-sample boolean tensor, which
makes modality dropout correct when a batch mixes available and missing
modalities.
"""

from typing import Optional, Union
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

Availability = Union[bool, torch.Tensor, None]


def _availability_mask(value: Availability,
                       batch_size: int,
                       device: torch.device) -> torch.Tensor:
    """Normalize a boolean/tensor availability flag to shape (B,)."""
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if isinstance(value, bool):
        return torch.full((batch_size,), value, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, device=device).bool().reshape(-1)
    if mask.numel() == 1:
        mask = mask.expand(batch_size)
    if mask.numel() != batch_size:
        raise ValueError(
            f"Availability mask has {mask.numel()} elements, expected {batch_size}"
        )
    return mask


class ModalityAwareFusion(nn.Module):
    """
    Modality-aware fusion for RGB and IR features.

    Supports:
    - RGB + IR: both modalities contribute
    - RGB only: RGB embedding is adapted
    - IR only: IR embedding is adapted
    - Neither: zero embedding

    Uses learnable projections and gated fusion. Output dimension is fixed
    (512-D) regardless of missing modalities.
    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        """
        Initialize modality-aware fusion.

        Args:
            input_dim: Input feature dimension (2048 for ResNet50)
            hidden_dim: Hidden dimension for projections
            output_dim: Output fusion dimension (512)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projections for each modality
        self.rgb_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.ir_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Fusion to output dimension
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        # Single-modality adaptation layers
        self.rgb_adapt = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        self.ir_adapt = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        logger.info(
            f"Modality-aware fusion: input={input_dim}, hidden={hidden_dim}, "
            f"output={output_dim}"
        )

    def forward(self,
                rgb_features: Optional[torch.Tensor],
                ir_features: Optional[torch.Tensor] = None,
                rgb_available: Availability = True,
                ir_available: Availability = True) -> torch.Tensor:
        """
        Fuse RGB and IR features with per-sample modality awareness.

        Args:
            rgb_features: RGB features (B, input_dim) or None
            ir_features: IR features (B, input_dim) or None
            rgb_available: bool or (B,) tensor
            ir_available: bool or (B,) tensor

        Returns:
            Fused features (B, output_dim)
        """
        if rgb_features is not None:
            batch_size = rgb_features.shape[0]
            device = rgb_features.device
            dtype = rgb_features.dtype
        elif ir_features is not None:
            batch_size = ir_features.shape[0]
            device = ir_features.device
            dtype = ir_features.dtype
        else:
            raise ValueError("At least one of rgb_features or ir_features is required")

        rgb_mask = _availability_mask(rgb_available, batch_size, device)
        ir_mask = _availability_mask(ir_available, batch_size, device)

        # A modality cannot be used if its features were not supplied.
        if rgb_features is None:
            rgb_mask = torch.zeros_like(rgb_mask)
        if ir_features is None:
            ir_mask = torch.zeros_like(ir_mask)

        zero_hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        zero_out = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

        if rgb_features is not None:
            rgb_embed = self.rgb_projection(rgb_features)
        else:
            rgb_embed = zero_hidden
        rgb_embed = rgb_embed * rgb_mask.unsqueeze(1).to(dtype)

        if ir_features is not None:
            ir_embed = self.ir_projection(ir_features)
        else:
            ir_embed = zero_hidden
        ir_embed = ir_embed * ir_mask.unsqueeze(1).to(dtype)

        # Both modalities
        both_mask = rgb_mask & ir_mask
        if both_mask.any():
            fused_both = self.fusion_layer(torch.cat([rgb_embed, ir_embed], dim=1))
        else:
            fused_both = zero_out

        # RGB only
        rgb_only_mask = rgb_mask & ~ir_mask
        fused_rgb = self.rgb_adapt(rgb_embed) if rgb_only_mask.any() else zero_out

        # IR only
        ir_only_mask = ir_mask & ~rgb_mask
        fused_ir = self.ir_adapt(ir_embed) if ir_only_mask.any() else zero_out

        fused = torch.where(both_mask.unsqueeze(1), fused_both, zero_out)
        fused = torch.where(rgb_only_mask.unsqueeze(1), fused_rgb, fused)
        fused = torch.where(ir_only_mask.unsqueeze(1), fused_ir, fused)

        return fused

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim
