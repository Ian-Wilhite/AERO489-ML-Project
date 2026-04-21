"""
AERO 489 — Model 1: Linear Regression

Baseline scalar model. Fits an OLS linear model on the engineered feature set
(ENGINEERED_COLS, 7 features). Expected to underfit given the nonlinear
relationship between damage state and g_limit, but establishes a performance floor.

Default feature set: ENGINEERED_COLS (7)
Cross-validation: 10-fold CV R² reported alongside test metrics
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

from .base import WingModel


class LinearReg(WingModel):
    name = "linear_regression"

    def __init__(self, cv_folds: int = 10):
        self.cv_folds = cv_folds
        self.cv_r2_: float | None = None  # set during fit
        self._pipeline: Pipeline | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # TODO: build sklearn Pipeline: StandardScaler → LinearRegression
        # TODO: run cross_val_score with KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        #       store mean CV R² in self.cv_r2_
        # TODO: fit the pipeline on the full training set
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: return self._pipeline.predict(X)
        raise NotImplementedError
