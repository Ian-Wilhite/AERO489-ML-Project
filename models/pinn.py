"""
AERO 489 — Model 6: Physics-Informed Neural Network (PINN)

Same MLP architecture as Model 5, but the training loss adds a physics residual
term that encodes the linear-elastic beam relationship between average strain
and reaction force.

Physics constraint
------------------
In the elastic regime, the wing obeys Hooke's law:
    Total_RF ≈ K_eff × avg_strain

where K_eff is an effective structural stiffness that can be estimated from the
training data (slope of RF vs. avg_strain across all simulations).

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .base import WingModel
from .feedforward_nn import _MLP  # reuse the same network definition

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AIRCRAFT_MASS_KG = 16_500.0
G_ACCEL          = 9.81


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
        patience: int = 30,
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
        self._k_eff: float | None = None  # effective stiffness fit from training data
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    # Index of avg_strain_at_failure in the feature vector (ALL_SCALAR_COLS position 2)
    # This is used in the physics loss to extract the strain value from each batch.
    # TODO: confirm index matches ALL_SCALAR_COLS ordering in data_utils.py
    _AVG_STRAIN_IDX = 2

    def _estimate_k_eff(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Fit K_eff = RF / avg_strain from training data (OLS through origin).

        RF_train = y_train * AIRCRAFT_MASS_KG * G_ACCEL / 2
        avg_strain = X_train[:, _AVG_STRAIN_IDX]  (after scaling — use raw X)
        """
        # TODO: compute RF_train from y_train
        # TODO: solve  K_eff = sum(RF * avg_strain) / sum(avg_strain²)  (OLS no intercept)
        # TODO: return float K_eff
        raise NotImplementedError

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)

        # TODO: estimate K_eff from raw (unscaled) training data via _estimate_k_eff
        # TODO: fit StandardScaler on X_train; transform X_train
        # TODO: split off 10% validation set
        # TODO: wrap in TensorDataset + DataLoader
        # TODO: instantiate _MLP and optimiser (same as FeedforwardNN)
        # TODO: training loop:
        #       - forward pass → g_pred
        #       - data loss: MSE(g_pred, g_true)
        #       - physics loss:
        #           RF_pred     = g_pred * AIRCRAFT_MASS_KG * G_ACCEL / 2
        #           strain_from_batch = X_batch[:, _AVG_STRAIN_IDX]  # raw (pre-scaled?)
        #           physics_res = RF_pred / self._k_eff - strain_from_batch
        #           physics_loss = MSE(physics_res, zeros_like(physics_res))
        #       - total_loss = data_loss + self.lambda_physics * physics_loss
        #       - backward, step, early stopping (same as FeedforwardNN)
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: scale X, forward pass, return numpy (same as FeedforwardNN.predict)
        raise NotImplementedError
