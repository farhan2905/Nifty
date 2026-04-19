from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class FeatureBlockEncoder(nn.Module):
    """Compress a feature block into a compact memory embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketMemoryAggregator(nn.Module):
    """Combine short-, mid-, and long-context vectors into one memory state."""

    def __init__(self, short_dim: int, mid_dim: int, long_dim: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.short = FeatureBlockEncoder(short_dim, out_dim // 2, dropout)
        self.mid = FeatureBlockEncoder(mid_dim, out_dim // 2, dropout)
        self.long = FeatureBlockEncoder(long_dim, out_dim // 2, dropout)
        self.fuse = nn.Sequential(
            nn.Linear((out_dim // 2) * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, short_ctx: torch.Tensor, mid_ctx: torch.Tensor, long_ctx: torch.Tensor) -> torch.Tensor:
        parts = [self.short(short_ctx), self.mid(mid_ctx), self.long(long_ctx)]
        return self.fuse(torch.cat(parts, dim=-1))
