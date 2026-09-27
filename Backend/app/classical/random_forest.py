"""Random Forest baseline model for Q-BioForge."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from app.classical.base import ClassicalModel


class RandomForestModel(ClassicalModel):
    """Random Forest ensemble classifier baseline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name="random_forest", config=config or {})

        n_estimators = self.config.get("n_estimators", 100)
        max_depth = self.config.get("max_depth", None)
        min_samples_split = self.config.get("min_samples_split", 2)
        min_samples_leaf = self.config.get("min_samples_leaf", 1)
        criterion = self.config.get("criterion", "gini")
        random_state = self.config.get("random_seed", self.config.get("random_state", 42))

        self.model_instance = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth is not None else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            criterion=criterion,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestModel":
        """Fit Random Forest model on training split."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
        self.model_instance.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestModel must be fitted before predict.")
        return self.model_instance.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestModel must be fitted before predict_proba.")
        return self.model_instance.predict_proba(x)
