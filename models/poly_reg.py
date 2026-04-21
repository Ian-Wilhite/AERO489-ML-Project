"""
AERO 489 — Model 2: Polynomial Regression

Extends the linear baseline by expanding the engineered feature set with
polynomial interaction terms. Degree is a hyperparameter; degree=2 is the
default since the g_limit–strain relationship is approximately quadratic
(strain energy ∝ strain², g_limit ∝ RF ∝ strain in elastic regime).

Default feature set : ENGINEERED_COLS (7)
Default degree      : 2
Cross-validation    : 10-fold CV R² reported alongside test metrics
Regularisation      : Ridge (α tuned via CV) to control overfitting from
                      the large number of interaction terms at degree ≥ 2
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

from .base import WingModel


class PolyReg(WingModel):
    name = "polynomial_regression"

    def __init__(self, degree: int = 2, alpha: float = 1.0, cv_folds: int = 10):
        self.degree = degree
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.cv_r2_: float | None = None
        self._pipeline: Pipeline | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # TODO: build Pipeline: StandardScaler → PolynomialFeatures(degree, include_bias=False)
        #       → StandardScaler → Ridge(alpha=self.alpha)
        # TODO: run cross_val_score; store mean CV R² in self.cv_r2_
        # TODO: fit on full training set
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: return self._pipeline.predict(X)
        raise NotImplementedError
