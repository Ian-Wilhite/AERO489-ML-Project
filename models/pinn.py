"""
AERO 489 — Model 6: Physics-Informed Neural Network (PINN)

Same MLP as FeedforwardNN + a normalised physics residual term in the loss.

Three physics models (select via physics_model argument):
  "hooke_strain"  — RF = K1 × avg_strain          (linear elastic, Hooke's law)
  "energy_quad"   — RF² = K2 × strain_energy       (U ∝ F² in elastic regime)
  "energy_rate"   — RF = K3 × strain_energy_slope  (Castigliano: dU/dF = tip_deflection ∝ RF/k)

Each residual is normalised by the training-set std of the reference feature so
that lambda is a dimensionless weight comparable across models:

  physics_loss = mean( ((residual) / feat_std)² )
  total_loss   = data_mse + lambda_physics × physics_loss

Feature indices in ALL_SCALAR_COLS (ENGINEERED_COLS block):
  2 → avg_strain_at_failure
  4 → strain_energy_at_failure
  5 → strain_energy_slope
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import WingModel
from .feedforward_nn import _MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AIRCRAFT_MASS_KG = 16_500.0
G_ACCEL          = 9.81
_HALF_WEIGHT     = AIRCRAFT_MASS_KG * G_ACCEL / 2.0   # converts g_limit → RF [N]

# Feature index within ALL_SCALAR_COLS for each physics model
_PHYS_CFG = {
    "hooke_strain": 2,   # avg_strain_at_failure
    "energy_quad":  4,   # strain_energy_at_failure
    "energy_rate":  5,   # strain_energy_slope
}


class PINN(WingModel):
    name = "pinn"

    def __init__(
        self,
        hidden: list[int] = [256, 128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_physics: float = 0.1,
        physics_model: str = "hooke_strain",
        epochs: int = 500,
        batch_size: int = 64,
        patience: int = 40,
        seed: int = 42,
    ):
        if physics_model not in _PHYS_CFG:
            raise ValueError(f"physics_model must be one of {list(_PHYS_CFG)}")
        self.hidden         = hidden
        self.dropout        = dropout
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.lambda_physics = lambda_physics
        self.physics_model  = physics_model
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.patience       = patience
        self.seed           = seed
        self._scaler: StandardScaler | None = None
        self._model: _MLP | None = None
        self._k:     float | None = None   # physics constant estimated from training data
        self._feat_scale: float | None = None  # normalisation denominator
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    # ── Physics setup ─────────────────────────────────────────────────────────

    def _setup_physics(self, X_raw: np.ndarray, y_train: np.ndarray) -> None:
        """Estimate the physics constant K and normalisation scale from training data."""
        feat_idx = _PHYS_CFG[self.physics_model]
        RF       = y_train * _HALF_WEIGHT
        feat     = X_raw[:, feat_idx]

        if self.physics_model == "hooke_strain":
            # RF = K × avg_strain  →  K = Σ(RF·ε) / Σ(ε²)  (OLS through origin)
            self._k = float(np.dot(RF, feat) / np.dot(feat, feat))
            residual = RF / self._k - feat

        elif self.physics_model == "energy_quad":
            # RF² = K × strain_energy  →  K = Σ(RF²·U) / Σ(U²)
            self._k = float(np.dot(RF**2, feat) / np.dot(feat, feat))
            residual = RF**2 / self._k - feat

        elif self.physics_model == "energy_rate":
            # RF = K × dU/dRF  →  K = Σ(RF·slope) / Σ(slope²)
            self._k = float(np.dot(RF, feat) / np.dot(feat, feat))
            residual = RF / self._k - feat

        self._feat_scale = float(np.std(feat)) + 1e-12
        print(f"    [{self.physics_model}]  K={self._k:.3e}  "
              f"residual_std={np.std(residual):.3e}  feat_std={self._feat_scale:.3e}")

    def _phys_residual(
        self,
        g_pred: torch.Tensor,
        feat_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Normalised physics residual for the current batch."""
        RF = g_pred * _HALF_WEIGHT
        k  = self._k
        s  = self._feat_scale

        if self.physics_model == "hooke_strain":
            return (RF / k - feat_batch) / s
        elif self.physics_model == "energy_quad":
            return (RF**2 / k - feat_batch) / s
        else:  # energy_rate
            return (RF / k - feat_batch) / s

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        self._setup_physics(X_train, y_train)

        feat_idx      = _PHYS_CFG[self.physics_model]
        raw_feat      = X_train[:, feat_idx].copy()

        self._scaler  = StandardScaler()
        X_scaled      = self._scaler.fit_transform(X_train)

        idx = np.arange(len(y_train))
        idx_tr, idx_val = train_test_split(idx, test_size=0.1, random_state=self.seed)

        Xt  = torch.tensor(X_scaled[idx_tr],  dtype=torch.float32, device=DEVICE)
        yt  = torch.tensor(y_train[idx_tr],   dtype=torch.float32, device=DEVICE)
        ft  = torch.tensor(raw_feat[idx_tr],  dtype=torch.float32, device=DEVICE)
        Xv  = torch.tensor(X_scaled[idx_val], dtype=torch.float32, device=DEVICE)
        yv  = torch.tensor(y_train[idx_val],  dtype=torch.float32, device=DEVICE)

        loader  = DataLoader(TensorDataset(Xt, ft, yt), batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()

        self._model = _MLP(X_train.shape[1], self.hidden, self.dropout).to(DEVICE)
        opt   = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for Xb, fb, yb in loader:
                opt.zero_grad()
                g_pred    = self._model(Xb)
                data_loss = loss_fn(g_pred, yb)
                phys_loss = (self._phys_residual(g_pred, fb) ** 2).mean()
                loss      = data_loss + self.lambda_physics * phys_loss
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(yb)
            sched.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(Xv), yv).item()

            self.train_loss_history_.append(train_loss / len(idx_tr))
            self.val_loss_history_.append(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            if (epoch + 1) % 50 == 0:
                print(f"    ep {epoch+1:4d}  train={self.train_loss_history_[-1]:.5f}  val={val_loss:.5f}")

        if best_state:
            self._model.load_state_dict(best_state)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            return self._model(Xt).cpu().numpy()
