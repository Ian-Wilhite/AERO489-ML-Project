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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .base import WingModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()
        # TODO: build nn.Sequential from n_features → hidden[0] → ... → hidden[-1] → 1
        #       with ReLU activations and Dropout(dropout) between hidden layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return self.net(x).squeeze(-1)
        raise NotImplementedError


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
        patience: int = 30,
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

        # TODO: fit StandardScaler on X_train; transform X_train
        # TODO: split off 10% of training data as an internal validation set
        #       (use train_test_split with random_state=self.seed)
        # TODO: wrap numpy arrays in TensorDataset + DataLoader
        # TODO: instantiate _MLP(n_features, self.hidden, self.dropout).to(DEVICE)
        # TODO: set up Adam optimiser + CosineAnnealingLR scheduler
        # TODO: training loop:
        #       - forward pass, MSELoss, backward, step optimiser + scheduler
        #       - evaluate val loss every epoch; append to history lists
        #       - implement early stopping: if val_loss hasn't improved for
        #         self.patience epochs, restore best weights and stop
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: scale X with self._scaler
        # TODO: convert to torch.Tensor, run self._model.eval() forward pass
        # TODO: return numpy array of predictions
        raise NotImplementedError
