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
from sklearn.pipeline import Pipeline
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
        # StandardScaler prevents float32 overflow for Box-Cox features with
        # extreme numerical ranges (e.g. boxcox_k_spring ∈ [5e33, 1e39]).
        # RF splits are rank-based so scaling does not change the tree structure.
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.seed),
            self.param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(X_scaled, y_train)
        self.best_params_ = grid_search.best_params_

        best_rf = RandomForestRegressor(**self.best_params_, random_state=self.seed)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(best_rf, X_scaled, y_train, cv=cv, scoring="r2")
        self.cv_r2_ = float(scores.mean())

        self._model = RandomForestRegressor(**self.best_params_, random_state=self.seed)
        self._model.fit(X_scaled, y_train)
        self.feature_importances_ = self._model.feature_importances_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))
