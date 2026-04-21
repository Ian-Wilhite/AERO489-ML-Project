"""
AERO 489 — Model 4: Random Forest Regression

Ensemble tree method — handles nonlinearity and feature interactions without
explicit polynomial expansion. Provides feature importances as a bonus
interpretability output.

Default feature set    : ALL_SCALAR_COLS (31: 7 engineered + 24 raw gauges)
Hyperparameter tuning  : GridSearchCV over n_estimators and max_depth
Cross-validation       : 5-fold inner CV for tuning; 10-fold outer CV for R² report
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

from .base import WingModel


# Default hyperparameter search grid
PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth":    [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}


class RandomForest(WingModel):
    name = "random_forest"

    def __init__(self, param_grid: dict = PARAM_GRID, cv_folds: int = 10, seed: int = 42):
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.seed = seed
        self.cv_r2_: float | None = None
        self.best_params_: dict | None = None
        self.feature_importances_: np.ndarray | None = None
        self._model: RandomForestRegressor | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # TODO: run GridSearchCV(RandomForestRegressor(random_state=self.seed),
        #           self.param_grid, cv=5, scoring="r2", n_jobs=-1)
        #       store best_params_ in self.best_params_
        # TODO: run outer cross_val_score with best params; store mean in self.cv_r2_
        # TODO: refit on full training set; store model in self._model
        # TODO: store self._model.feature_importances_ in self.feature_importances_
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: return self._model.predict(X)
        raise NotImplementedError
