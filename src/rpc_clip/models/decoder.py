from __future__ import annotations

import torch
from torch import nn


class PatchDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        batch, _, dim = tokens.shape
        grid_h, grid_w = grid_hw
        x = tokens.transpose(1, 2).reshape(batch, dim, grid_h, grid_w)
        return self.net(x)

