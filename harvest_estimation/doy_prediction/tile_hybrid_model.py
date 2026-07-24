from __future__ import annotations

import torch
from torch import nn

from doy_prediction.tile_cnn_model import conv_block
from doy_prediction.tile_rnn_model import resolve_rnn_class


class TileCNNRNNHybridRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        cnn_hidden_channels: tuple[int, int, int] | list[int] = (32, 64, 128),
        rnn_hidden_size: int = 64,
        rnn_num_layers: int = 2,
        rnn_hidden_sizes: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if len(cnn_hidden_channels) != 3:
            raise ValueError("cnn_hidden_channels must contain exactly three values")
        c1, c2, c3 = tuple(cnn_hidden_channels)
        self.cnn_features = nn.Sequential(
            conv_block(in_channels, c1, dropout),
            conv_block(c1, c2, dropout),
            conv_block(c2, c3, dropout),
        )
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.start_head = nn.Sequential(
            nn.Linear(c3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        if rnn_hidden_sizes is not None:
            rnn_hidden_sizes = tuple(rnn_hidden_sizes)
            if not rnn_hidden_sizes:
                raise ValueError("rnn_hidden_sizes must contain at least one layer size")
            if any(size < 1 for size in rnn_hidden_sizes):
                raise ValueError("rnn_hidden_sizes values must be >= 1")
        elif rnn_num_layers < 1:
            raise ValueError("rnn_num_layers must be >= 1")

        self.rnn_hidden_sizes = rnn_hidden_sizes
        self.bidirectional = bidirectional
        rnn_cls = resolve_rnn_class(rnn_type)
        directions = 2 if bidirectional else 1

        if rnn_hidden_sizes is None:
            self.rnn = rnn_cls(
                input_size=in_channels,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=dropout if rnn_num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            rnn_head_in_features = rnn_hidden_size * directions
        else:
            self.rnn_layers = nn.ModuleList()
            layer_input_size = in_channels
            for layer_hidden_size in rnn_hidden_sizes:
                self.rnn_layers.append(
                    rnn_cls(
                        input_size=layer_input_size,
                        hidden_size=layer_hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=0.0,
                        bidirectional=bidirectional,
                    )
                )
                layer_input_size = layer_hidden_size * directions
            self.recurrent_dropout = nn.Dropout(dropout)
            rnn_head_in_features = rnn_hidden_sizes[-1] * directions

        self.end_head = nn.Sequential(
            nn.Linear(rnn_head_in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, seq_len), got {tuple(x.shape)}")

        cnn_features = self.cnn_features(x)
        cnn_pooled = self.cnn_pool(cnn_features).squeeze(-1)
        start = self.start_head(cnn_pooled).squeeze(-1)

        rnn_features = self._rnn_features(x)
        end = self.end_head(rnn_features).squeeze(-1)
        return torch.stack((start, end), dim=1)

    def _rnn_features(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.transpose(1, 2)
        if self.rnn_hidden_sizes is not None:
            output = x_seq
            hidden = None
            for layer_idx, layer in enumerate(self.rnn_layers):
                output, hidden = layer(output)
                if layer_idx < len(self.rnn_layers) - 1:
                    output = self.recurrent_dropout(output)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            num_directions = 2 if self.bidirectional else 1
            last_hidden_size = self.rnn_hidden_sizes[-1]
            hidden = hidden.view(1, num_directions, x.shape[0], last_hidden_size)
            return hidden[-1].transpose(0, 1).reshape(x.shape[0], last_hidden_size * num_directions)

        _, hidden = self.rnn(x_seq)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        num_directions = 2 if self.rnn.bidirectional else 1
        hidden = hidden.view(self.rnn.num_layers, num_directions, x.shape[0], self.rnn.hidden_size)
        return hidden[-1].transpose(0, 1).reshape(x.shape[0], self.rnn.hidden_size * num_directions)
