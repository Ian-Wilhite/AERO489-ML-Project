"""
AERO 489 — Model 6: Physics-Informed Neural Network (PINN)

Same MLP architecture as Model 5, but the training loss adds a physics residual
term that encodes the linear-elastic beam relationship between average strain
and reaction force.

Physics constraint
------------------
In the elastic regime, the wing obeys Hooke's law:
    Total_RF ≈ K_eff × avg_strain

where K_eff is an effective structural stiffness estimated from training data
(OLS through origin: K_eff = Σ(RF·ε) / Σ(ε²)).

The g_limit predicted by the network implies a failure force:
    RF_pred = g_pred × AIRCRAFT_MASS_KG × G_ACCEL / 2

The physics residual penalises deviation from:
    RF_pred ≈ K_eff × avg_strain_at_failure

Loss = MSE(g_pred, g_true) + λ_physics × MSE(RF_pred / K_eff, avg_strain)

λ_physics is a hyperparameter (default 0.1); set to 0 to recover Model 5.

Architecture       : same as FeedforwardNN (input → [256, 128, 64] → 1)
Default feature set: ALL_SCALAR_COLS (31)
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

# Index of avg_strain_at_failure in ALL_SCALAR_COLS (ENGINEERED_COLS position 2)
_AVG_STRAIN_IDX = 2


class PINN(WingModel):
    name = "pinn"

    def __init__(
        self,
        hidden: list[int] = [256, 128, 64],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_physics: float = 0.1,
        epochs: int = 500,
        batch_size: int = 64,
        patience: int = 40,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_physics = lambda_physics
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model: _MLP | None = None
        self._k_eff: float | None = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def _estimate_k_eff(self, X_raw: np.ndarray, y_train: np.ndarray) -> float:
        """OLS through origin: K_eff = Σ(RF·ε) / Σ(ε²) from training data."""
        RF         = y_train * AIRCRAFT_MASS_KG * G_ACCEL / 2.0
        avg_strain = X_raw[:, _AVG_STRAIN_IDX]
        return float(np.dot(RF, avg_strain) / np.dot(avg_strain, avg_strain))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        # Capture K_eff and raw avg_strain BEFORE scaling
        self._k_eff    = self._estimate_k_eff(X_train, y_train)
        raw_avg_strain = X_train[:, _AVG_STRAIN_IDX].copy()
        print(f"    K_eff = {self._k_eff:.3e} N/m  (λ_physics={self.lambda_physics})")

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        # Val split — keep raw_avg_strain aligned via index split
        idx = np.arange(len(y_train))
        idx_tr, idx_val = train_test_split(idx, test_size=0.1, random_state=self.seed)

        Xt  = torch.tensor(X_scaled[idx_tr],      dtype=torch.float32, device=DEVICE)
        yt  = torch.tensor(y_train[idx_tr],        dtype=torch.float32, device=DEVICE)
        st  = torch.tensor(raw_avg_strain[idx_tr], dtype=torch.float32, device=DEVICE)
        Xv  = torch.tensor(X_scaled[idx_val],      dtype=torch.float32, device=DEVICE)
        yv  = torch.tensor(y_train[idx_val],        dtype=torch.float32, device=DEVICE)

        loader  = DataLoader(TensorDataset(Xt, st, yt), batch_size=self.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        k_t     = torch.tensor(self._k_eff, dtype=torch.float32, device=DEVICE)
        scale   = float(AIRCRAFT_MASS_KG * G_ACCEL / 2.0)

        self._model = _MLP(X_train.shape[1], self.hidden, self.dropout).to(DEVICE)
        opt   = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for Xb, sb, yb in loader:
                opt.zero_grad()
                g_pred    = self._model(Xb)
                data_loss = loss_fn(g_pred, yb)
                RF_pred   = g_pred * scale
                phys_res  = RF_pred / k_t - sb
                phys_loss = (phys_res ** 2).mean()
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
