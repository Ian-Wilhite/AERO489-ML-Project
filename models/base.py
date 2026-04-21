"""
AERO 489 — Abstract base class for all wing g-limit prediction models.

Every model in models/ must subclass WingModel and implement fit() and predict().
The evaluate() method is provided here and calls evaluate.score() so metric
computation stays in one place.
"""

from abc import ABC, abstractmethod
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import evaluate as ev


class WingModel(ABC):
    """Abstract interface for all g-limit prediction models."""

    name: str = "unnamed"  # override in each subclass for results/ filenames

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the provided feature matrix and targets."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted g_limit values for each row of X."""
        ...

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_features: int,
        n_inference_reps: int = 100,
    ) -> dict:
        """Compute all six proposal metrics on the held-out test set.

        Returns a dict suitable for JSON serialisation and compare.py.
        Keys: adj_r2, mae, rmse, max_overpredict, mos_01, inference_time_ms
        """
        return ev.score(self, X_test, y_test, n_features, n_inference_reps)
