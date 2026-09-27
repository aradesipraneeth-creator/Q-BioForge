"""Support Vector Machine baseline model for Q-BioForge."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.svm import SVC
from app.classical.base import ClassicalModel


class SVMModel(ClassicalModel):
    """Support Vector Machine (SVC) classifier baseline with probabilistic calibration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name="svm", config=config or {})

        c_param = self.config.get("C", 1.0)
        kernel = self.config.get("kernel", "rbf")
        gamma = self.config.get("gamma", "scale")
        probability = self.config.get("probability", True)
        random_state = self.config.get("random_seed", self.config.get("random_state", 42))

        self.model_instance = SVC(
            C=float(c_param),
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "SVMModel":
        """Fit SVM model on training split."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
        self.model_instance.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels."""
        if not self.is_fitted:
            raise RuntimeError("SVMModel must be fitted before predict.")
        return self.model_instance.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict calibrated class probabilities via Platt scaling."""
        if not self.is_fitted:
            raise RuntimeError("SVMModel must be fitted before predict_proba.")
        return self.model_instance.predict_proba(x)
