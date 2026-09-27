"""Logistic Regression baseline model for Q-BioForge."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from app.classical.base import ClassicalModel


class LogisticRegressionModel(ClassicalModel):
    """Logistic Regression baseline classifier with l2 regularization and probabilistic outputs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name="logistic_regression", config=config or {})
        
        c_param = self.config.get("C", 1.0)
        solver = self.config.get("solver", "lbfgs")
        max_iter = self.config.get("max_iter", 1000)
        tol = self.config.get("tol", 1e-4)
        random_state = self.config.get("random_seed", self.config.get("random_state", 42))

        kwargs = {
            "C": float(c_param),
            "solver": solver,
            "max_iter": int(max_iter),
            "tol": float(tol),
            "random_state": random_state,
        }
        if "penalty" in self.config and self.config["penalty"] != "l2":
            kwargs["penalty"] = self.config["penalty"]

        self.model_instance = LogisticRegression(**kwargs)


    def fit(self, x_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "LogisticRegressionModel":
        """Fit Logistic Regression model on training split."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
        self.model_instance.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionModel must be fitted before predict.")
        return self.model_instance.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict calibrated class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionModel must be fitted before predict_proba.")
        return self.model_instance.predict_proba(x)
