"""
AERO 489 — Model 5: Feedforward Neural Network (MLP)

Fully connected PyTorch network mapping scalar features → g_limit.
Acts as the baseline "modern ML" model before physics constraints (Model 6)
or temporal structure (Model 7) are added.

Architecture       : input → [256, 128, 64] → 1 (ReLU hidden, linear output)
Default feature set: ALL_SCALAR_COLS (31)
Loss               : MSE
Optimiser          : Adam with cosine-annealing LR schedule
Regularisation     : Dropout(0.2) + weight decay
Training           : early stopping on validation loss (held-out 10% of train)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import WingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FeedforwardNN(WingModel):
    name = "feedforward_nn"

    def __init__(
        self,
        hidden: list[int] = [256, 128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 500,
        batch_size: int = 64,
        patience: int = 40,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: _MLP | None = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_train, test_size=0.1, random_state=self.seed
        )

        Xt = torch.tensor(X_tr,  dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y_tr,  dtype=torch.float32, device=DEVICE)
        Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        yv = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        self._model = _MLP(X_train.shape[1], self.hidden, self.dropout).to(DEVICE)
        opt   = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for Xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self._model(Xb), yb)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(yb)
            sched.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(Xv), yv).item()

            self.train_loss_history_.append(train_loss / len(y_tr))
            self.val_loss_history_.append(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"    early stop epoch {epoch+1}  best_val={best_val:.5f}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"    ep {epoch+1:4d}  train={self.train_loss_history_[-1]:.5f}  val={val_loss:.5f}")

        if best_state:
            self._model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            return self._model(Xt).cpu().numpy()
