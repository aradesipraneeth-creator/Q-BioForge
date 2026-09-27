"""XGBoost gradient boosting baseline model for Q-BioForge."""

from typing import Any, Dict, List, Optional
import numpy as np
from app.classical.base import ClassicalModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel(ClassicalModel):
    """XGBoost extreme gradient boosting classifier baseline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name="xgboost", config=config or {})

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed in the environment. "
                "Please run 'pip install xgboost' or verify requirements.txt."
            )

        n_estimators = self.config.get("n_estimators", 100)
        max_depth = self.config.get("max_depth", 3)
        learning_rate = self.config.get("learning_rate", 0.1)
        subsample = self.config.get("subsample", 1.0)
        eval_metric = self.config.get("eval_metric", "logloss")
        random_state = self.config.get("random_seed", self.config.get("random_state", 42))

        self.model_instance = xgb.XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            eval_metric=eval_metric,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "XGBoostModel":
        """Fit XGBoost model on training split."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
        self.model_instance.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostModel must be fitted before predict.")
        return self.model_instance.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostModel must be fitted before predict_proba.")
        return self.model_instance.predict_proba(x)
