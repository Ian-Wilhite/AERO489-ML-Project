"""
AERO 489 — Model 7: Deep Learning — Bidirectional LSTM on load-step sequences

Unlike Models 1–6 which consume only failure-point features, this model ingests
the full loading history (all load steps from 0 → failure) and learns to predict
g_limit from how the strain field and tip deflection evolve under increasing load.

Input  : time_series.parquet — shape (n_sims, MAX_SEQ_LEN=29, 26 channels)
         Channels: 24 strain gauges + Tip_Deflection + load_fraction
         Sequences are zero-padded; actual lengths passed as seq_lens.
Output : g_limit scalar per simulation

Architecture  : bidirectional LSTM (2 layers, hidden=128) → Linear(256, 1)
                Final hidden state = concat(fwd_last, bwd_first) via h_n.
Loss          : MSE
Optimiser     : Adam + ReduceLROnPlateau
Regularisation: LSTM dropout + early stopping
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .base import WingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, seq_lens: np.ndarray, y: np.ndarray):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.long)
        self.y        = torch.tensor(y,        dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.seq_lens[idx], self.y[idx]


class _LSTMModel(nn.Module):
    def __init__(self, n_channels: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_channels, hidden, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (n_layers*2, batch, hidden) — last layer fwd [-2] and bwd [-1]
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(h).squeeze(-1)


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
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.seed       = seed
        self._model: _LSTMModel | None = None
        self._channel_mean: np.ndarray | None = None
        self._channel_std:  np.ndarray | None = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._channel_mean) / self._channel_std

    def fit(self, X_train: np.ndarray, seq_lens_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        # Per-channel stats from non-padded steps only
        mask  = np.zeros(X_train.shape[:2], dtype=bool)
        for i, l in enumerate(seq_lens_train):
            mask[i, :l] = True
        valid = X_train[mask]  # (total_steps, n_channels)
        self._channel_mean = valid.mean(axis=0, keepdims=True)
        self._channel_std  = valid.std(axis=0, keepdims=True) + 1e-8

        X_norm = self._normalise(X_train)

        idx = np.arange(len(y_train))
        idx_tr, idx_val = train_test_split(idx, test_size=0.1, random_state=self.seed)

        train_ds = _SeqDataset(X_norm[idx_tr],  seq_lens_train[idx_tr],  y_train[idx_tr])
        val_ds   = _SeqDataset(X_norm[idx_val], seq_lens_train[idx_val], y_train[idx_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=len(val_ds))

        n_channels  = X_train.shape[2]
        self._model = _LSTMModel(n_channels, self.hidden, self.n_layers, self.dropout).to(DEVICE)
        opt         = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        sched       = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
        loss_fn     = nn.MSELoss()

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for Xb, lb, yb in train_loader:
                Xb, lb, yb = Xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self._model(Xb, lb), yb)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(yb)

            self._model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for Xb, lb, yb in val_loader:
                    Xb, lb, yb = Xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
                    val_loss += loss_fn(self._model(Xb, lb), yb).item() * len(yb)
                val_loss /= len(val_ds)

            self.train_loss_history_.append(train_loss / len(train_ds))
            self.val_loss_history_.append(val_loss)
            sched.step(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"    early stop epoch {epoch+1}  best_val={best_val:.5f}")
                    break

            if (epoch + 1) % 25 == 0:
                print(f"    ep {epoch+1:3d}  train={self.train_loss_history_[-1]:.5f}  val={val_loss:.5f}")

        if best_state:
            self._model.load_state_dict(best_state)

    def predict(self, X: np.ndarray, seq_lens: np.ndarray | None = None) -> np.ndarray:
        if seq_lens is None:
            seq_lens = np.array(
                [max(1, int((X[i] != 0).any(axis=1).sum())) for i in range(len(X))],
                dtype=np.int64,
            )
        X_norm = self._normalise(X)
        ds     = _SeqDataset(X_norm, seq_lens, np.zeros(len(X_norm), dtype=np.float32))
        loader = DataLoader(ds, batch_size=64)
        preds  = []
        self._model.eval()
        with torch.no_grad():
            for Xb, lb, _ in loader:
                Xb, lb = Xb.to(DEVICE), lb.to(DEVICE)
                preds.append(self._model(Xb, lb).cpu().numpy())
        return np.concatenate(preds)
