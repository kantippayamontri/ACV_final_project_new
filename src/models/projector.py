"""Visual projector: maps VideoMAE features to LLM embedding space."""
import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """N-layer MLP with GELU that maps [B, T, D_in] -> [B, T, D_out]."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, output_dim)]
        for _ in range(1, num_layers):
            layers.append(nn.GELU())
            layers.append(nn.Linear(output_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = x.float()
        x = self.net(x)
        if mask is not None:
            return x, mask
        return x
