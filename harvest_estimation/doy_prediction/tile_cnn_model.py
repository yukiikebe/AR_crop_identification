from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TileCNNRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        c1, c2, c3 = hidden_channels
        self.features = nn.Sequential(
            conv_block(in_channels, c1, dropout),
            conv_block(c1, c2, dropout),
            conv_block(c2, c3, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


def normalized_to_doy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    doy = np.clip(np.rint(np.asarray(values, dtype=np.float32) * 366.0), 1, 366).astype(np.int32)
    doy[:, 1] = np.maximum(doy[:, 0], doy[:, 1])
    return doy
