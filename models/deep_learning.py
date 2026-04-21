"""
AERO 489 — Model 7: Deep Learning — LSTM on load-step sequences

Unlike Models 1–6 which consume only failure-point features, this model ingests
the full loading history (all load steps from 0 → failure) and learns to predict
g_limit from how the strain field and tip deflection evolve under increasing load.
This mirrors the real deployment scenario: the pilot observes the sensor trace
during a pull-up manoeuvre and the model extrapolates to the failure point.

Input  : time_series.parquet — shape (n_sims, MAX_SEQ_LEN=29, 26 channels)
         Channels: 24 strain gauges + Tip_Deflection + load_fraction
         Sequences are zero-padded; actual lengths stored in seq_lens for pack_padded.
Output : g_limit scalar per simulation

Architecture  : bidirectional LSTM (2 layers, hidden=128) → Linear(128, 1)
                PackedSequence used to handle variable-length histories.
Alternative   : 1D-CNN (Conv1d → GlobalAvgPool → Linear) — try if LSTM overfits
Loss          : MSE
Optimiser     : Adam + ReduceLROnPlateau
Regularisation: LSTM dropout + early stopping
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .base import WingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _SeqDataset(Dataset):
    """Wraps padded sequence arrays into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, seq_lens: np.ndarray, y: np.ndarray):
        # X      : (n_sims, MAX_SEQ_LEN, n_channels)
        # seq_lens: (n_sims,) — actual sequence length before padding
        # y      : (n_sims,)
        self.X        = torch.tensor(X, dtype=torch.float32)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.long)
        self.y        = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.seq_lens[idx], self.y[idx]


class _LSTMModel(nn.Module):
    def __init__(self, n_channels: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        # TODO: define bidirectional LSTM:
        #   self.lstm = nn.LSTM(n_channels, hidden, n_layers,
        #                       batch_first=True, bidirectional=True,
        #                       dropout=dropout if n_layers > 1 else 0)
        # TODO: define output head:
        #   self.head = nn.Linear(hidden * 2, 1)  (* 2 for bidirectional)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        # TODO: pack sequences: pack_padded_sequence(x, seq_lens.cpu(),
        #           batch_first=True, enforce_sorted=False)
        # TODO: run through self.lstm → take hidden state of last real step
        #       (use packed_output, unpack, index by seq_lens - 1)
        # TODO: pass through self.head, return squeezed output
        raise NotImplementedError


class DeepLearning(WingModel):
    name = "deep_learning_lstm"

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 25,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self._model: _LSTMModel | None = None
        self._channel_mean: np.ndarray | None = None  # for per-channel normalisation
        self._channel_std:  np.ndarray | None = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Per-channel z-score using training statistics."""
        # TODO: return (X - self._channel_mean) / (self._channel_std + 1e-8)
        raise NotImplementedError

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Parameters
        ----------
        X_train : np.ndarray  shape (n_sims, MAX_SEQ_LEN, n_channels)
        y_train : np.ndarray  shape (n_sims,)

        Note: caller (train_modern.py) must provide seq_lens alongside X_train.
        Consider changing the signature to fit(X_train, seq_lens_train, y_train)
        or packing seq_lens as the last channel.
        """
        torch.manual_seed(self.seed)

        # TODO: compute per-channel mean/std from non-padded steps only; store as
        #       self._channel_mean, self._channel_std for use in _normalise
        # TODO: normalise X_train
        # TODO: derive seq_lens from X_train (count non-zero rows per sim, or
        #       receive as argument — decide interface and document it)
        # TODO: split off ~10% validation sims
        # TODO: build _SeqDataset + DataLoader for train and val
        # TODO: instantiate _LSTMModel(n_channels, hidden, n_layers, dropout).to(DEVICE)
        # TODO: Adam optimiser + ReduceLROnPlateau(factor=0.5, patience=10)
        # TODO: training loop with early stopping (same pattern as FeedforwardNN)
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: normalise X with stored channel stats
        # TODO: derive seq_lens (same logic as fit)
        # TODO: batch forward pass in eval mode; return numpy predictions
        raise NotImplementedError
