"""
AERO 489 — Model 3: Gaussian Process Regression (GPR)

GPR is well-suited to small/medium datasets (~380 training samples here) and
naturally produces calibrated uncertainty estimates alongside point predictions.
The uncertainty output can serve as a data-driven margin of safety.

Default feature set : ALL_SCALAR_COLS (31: 7 engineered + 24 raw gauges)
Kernel              : RBF × ConstantKernel + WhiteKernel (noise)
                      — start here; try Matern(nu=2.5) if RBF underfits
Normalisation       : StandardScaler before GPR (GPR is not scale-invariant)
Cross-validation    : 10-fold CV R²

Notes
-----
sklearn GaussianProcessRegressor scales O(n³) in training size.  At ~380
samples this is fine; if the dataset grows, switch to sparse GP (GPyTorch).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

from .base import WingModel


class GPR(WingModel):
    name = "gaussian_process_regression"

    def __init__(self, kernel=None, cv_folds: int = 10, n_restarts: int = 5):
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        self.kernel = kernel
        self.cv_folds = cv_folds
        self.n_restarts = n_restarts
        self.cv_r2_: float | None = None
        self._scaler: StandardScaler | None = None
        self._gpr: GaussianProcessRegressor | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        self._gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
        )
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self._gpr, X_scaled, y_train, cv=cv, scoring="r2")
        self.cv_r2_ = float(scores.mean())
        self._gpr.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._gpr.predict(X_scaled)

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) — std is the GPR posterior standard deviation.

        The std can be used as a data-driven, input-dependent margin of safety:
            conservative_g_limit = mean - k * std  (k typically 1–2)
        """
        X_scaled = self._scaler.transform(X)
        return self._gpr.predict(X_scaled, return_std=True)
