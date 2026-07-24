from __future__ import annotations

import torch
from torch import nn


class TileRNNRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        hidden_sizes: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if hidden_sizes is not None:
            hidden_sizes = tuple(hidden_sizes)
            if not hidden_sizes:
                raise ValueError("hidden_sizes must contain at least one layer size")
            if any(size < 1 for size in hidden_sizes):
                raise ValueError("hidden_sizes values must be >= 1")
        elif num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.in_channels = in_channels

        rnn_cls = resolve_rnn_class(rnn_type)
        self.rnn_type = rnn_type.lower()
        self.hidden_sizes = hidden_sizes
        self.bidirectional = bidirectional

        directions = 2 if bidirectional else 1
        if hidden_sizes is None:
            self.rnn = rnn_cls(
                input_size=in_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            head_in_features = hidden_size * directions
        else:
            self.rnn_layers = nn.ModuleList()
            layer_input_size = in_channels
            for layer_hidden_size in hidden_sizes:
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
            head_in_features = hidden_sizes[-1] * directions

        self.head = nn.Sequential(
            nn.Linear(head_in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, seq_len), got {tuple(x.shape)}")

        x_seq = x.transpose(1, 2)
        if self.hidden_sizes is not None:
            output = x_seq
            hidden = None
            for layer_idx, layer in enumerate(self.rnn_layers):
                output, hidden = layer(output)
                if layer_idx < len(self.rnn_layers) - 1:
                    output = self.recurrent_dropout(output)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

            num_directions = 2 if self.bidirectional else 1
            last_hidden_size = self.hidden_sizes[-1]
            hidden = hidden.view(1, num_directions, x.shape[0], last_hidden_size)
            return self.head(hidden[-1].transpose(0, 1).reshape(x.shape[0], last_hidden_size * num_directions))

        _, hidden = self.rnn(x_seq)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        num_directions = 2 if self.rnn.bidirectional else 1
        hidden = hidden.view(self.rnn.num_layers, num_directions, x.shape[0], self.rnn.hidden_size)
        last_layer = hidden[-1].transpose(0, 1).reshape(x.shape[0], self.rnn.hidden_size * num_directions)
        return self.head(last_layer)


def resolve_rnn_class(rnn_type: str) -> type[nn.RNNBase]:
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU
    if rnn_type == "lstm":
        return nn.LSTM
    if rnn_type == "rnn":
        return nn.RNN
    raise ValueError("rnn_type must be one of: gru, lstm, rnn")
