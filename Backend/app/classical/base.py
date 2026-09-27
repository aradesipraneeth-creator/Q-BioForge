"""Abstract base interface for Classical Machine Learning models in Q-BioForge.

Provides a unified interface:
- fit(x_train, y_train, feature_names)
- predict(x)
- predict_proba(x)
- save_checkpoint(filepath)
- load_checkpoint(filepath)
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import joblib


class ClassicalModel(ABC):
    """Base interface for classical ML models in Q-BioForge."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted: bool = False
        self.model_instance: Any = None
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "ClassicalModel":
        """Train model on preprocessed training split."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate discrete class predictions."""
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Generate predicted class probabilities (2D array [P(y=0), P(y=1)])."""
        pass

    def save_checkpoint(self, filepath: str) -> str:
        """Persist fitted model checkpoint and configuration using joblib."""
        if not self.is_fitted or self.model_instance is None:
            raise RuntimeError(f"Cannot save unfitted model '{self.model_name}'.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint_data = {
            "model_name": self.model_name,
            "config": self.config,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "model_instance": self.model_instance,
        }
        joblib.dump(checkpoint_data, filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> "ClassicalModel":
        """Load fitted model checkpoint from disk into this instance."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint_data = joblib.load(filepath)
        self.model_name = checkpoint_data["model_name"]
        self.config = checkpoint_data.get("config", {})
        self.feature_names = checkpoint_data.get("feature_names", [])
        self.is_fitted = checkpoint_data.get("is_fitted", False)
        self.model_instance = checkpoint_data.get("model_instance")
        return self
