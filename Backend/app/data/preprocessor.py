"""Data preprocessing pipeline with strict leakage prevention for biomedical tabular datasets.

Features:
- Train / Validation / Test stratified splitting
- Missing value imputation fitted exclusively on training set
- Scaling/normalization fitted exclusively on training set
- Dimensionality reduction and quantum-ready feature bounds (e.g., [0, pi] or [-pi, pi])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


class DataSplit(BaseModel):
    """Container for split dataset subsets."""
    train_ratio: float = Field(0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(0.15, ge=0.0, le=0.5)
    test_ratio: float = Field(0.15, ge=0.0, le=0.5)
    random_seed: int = 42
    stratify: bool = True


class PreprocessorConfig(BaseModel):
    """Configuration for data preprocessing."""
    impute_strategy: str = "median"  # mean, median, most_frequent
    scaling_method: str = "minmax"   # standard, minmax, robust, quantum_angle ([0, pi])
    target_feature_range: Tuple[float, float] = (0.0, np.pi)  # [0, pi] for angle encoding
    target_dim_for_quantum: Optional[int] = None  # Reduces features to match qubit count
    dim_reduction_method: str = "select_k_best"   # pca or select_k_best


class BaseDataPipeline(ABC):
    """Abstract data pipeline enforcing zero-leakage preprocessing."""

    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.is_fitted: bool = False
        self._fitted_transformers: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, x_train: Union[pd.DataFrame, np.ndarray], y_train: Optional[Union[pd.Series, np.ndarray]] = None) -> "BaseDataPipeline":
        """Fit preprocessor parameters ONLY on the training split."""
        pass

    @abstractmethod
    def transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply pre-fitted transformations to any split (train, val, or test)."""
        pass

    def fit_transform(self, x_train: Union[pd.DataFrame, np.ndarray], y_train: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
        """Fit on train and transform train."""
        return self.fit(x_train, y_train).transform(x_train)


class TabularPreprocessor(BaseDataPipeline):
    """Concrete zero-leakage preprocessing pipeline for classical and quantum biomedical ML."""

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        super().__init__(config or PreprocessorConfig())
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None
        self.dim_reducer: Optional[Union[PCA, SelectKBest]] = None
        self.feature_names_in: List[str] = []

    def fit(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "TabularPreprocessor":
        """Fit imputation, scaling, and feature reduction strictly on x_train (and y_train if supervised)."""
        if isinstance(x_train, pd.DataFrame):
            self.feature_names_in = list(x_train.columns)
            x_arr = x_train.to_numpy(dtype=np.float64)
        else:
            x_arr = np.asarray(x_train, dtype=np.float64)

        if y_train is not None:
            y_arr = np.asarray(y_train)
        else:
            y_arr = None

        # 1. Fit Imputer
        self.imputer = SimpleImputer(strategy=self.config.impute_strategy)
        x_imputed = self.imputer.fit_transform(x_arr)

        # 2. Fit Scaler
        if self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif self.config.scaling_method == "quantum_angle":
            self.scaler = MinMaxScaler(feature_range=self.config.target_feature_range)
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        x_scaled = self.scaler.fit_transform(x_imputed)

        # 3. Fit Dimensionality Reducer if target dimension is specified
        self.quantum_scaler = None
        if self.config.target_dim_for_quantum and self.config.target_dim_for_quantum < x_scaled.shape[1]:
            k = self.config.target_dim_for_quantum
            if self.config.dim_reduction_method == "pca":
                self.dim_reducer = PCA(n_components=k, random_state=42)
                x_reduced = self.dim_reducer.fit_transform(x_scaled)
            elif self.config.dim_reduction_method == "select_k_best":
                if y_arr is None:
                    raise ValueError("Supervised feature selection (select_k_best) requires y_train.")
                self.dim_reducer = SelectKBest(score_func=f_classif, k=k)
                x_reduced = self.dim_reducer.fit_transform(x_scaled, y_arr)
            else:
                x_reduced = x_scaled

            # Fit quantum angle scaler strictly on training reduced data
            if self.config.scaling_method == "quantum_angle":
                self.quantum_scaler = MinMaxScaler(feature_range=self.config.target_feature_range)
                self.quantum_scaler.fit(x_reduced)

        self.is_fitted = True
        return self

    def transform(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using pre-fitted transformers without altering state."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        if isinstance(x, pd.DataFrame):
            x_arr = x.to_numpy(dtype=np.float64)
        else:
            x_arr = np.asarray(x, dtype=np.float64)

        # 1. Impute
        x_imputed = self.imputer.transform(x_arr)

        # 2. Scale
        x_scaled = self.scaler.transform(x_imputed)

        # 3. Reduce dimension if configured
        if self.dim_reducer is not None:
            x_final = self.dim_reducer.transform(x_scaled)
            if self.config.scaling_method == "quantum_angle" and self.quantum_scaler is not None:
                x_final = self.quantum_scaler.transform(x_final)
            return x_final

        return x_scaled

    def get_transformed_feature_names(self) -> List[str]:
        """Return the precise feature names resulting from preprocessing."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to get transformed feature names.")

        if self.dim_reducer is None:
            if self.feature_names_in:
                return list(self.feature_names_in)
            return [f"feature_{i}" for i in range(self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 0)]

        if isinstance(self.dim_reducer, PCA):
            return [f"PCA_Component_{i+1}" for i in range(self.dim_reducer.n_components_)]
        elif isinstance(self.dim_reducer, SelectKBest):
            if self.feature_names_in:
                selected_indices = self.dim_reducer.get_support(indices=True)
                return [self.feature_names_in[i] for i in selected_indices]
            return [f"Selected_Feature_{i+1}" for i in range(self.dim_reducer.k)]

        return [f"Transformed_Feature_{i+1}" for i in range(self.config.target_dim_for_quantum or 0)]

    def get_transformation_metadata(self) -> Dict[str, Any]:
        """Return comprehensive transformation parameters and explainability metadata."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to retrieve transformation metadata.")

        meta: Dict[str, Any] = {
            "impute_strategy": self.config.impute_strategy,
            "scaling_method": self.config.scaling_method,
            "target_feature_range": list(self.config.target_feature_range),
            "input_features_count": len(self.feature_names_in) if self.feature_names_in else 0,
            "original_feature_names": list(self.feature_names_in) if self.feature_names_in else [],
            "output_features_count": self.config.target_dim_for_quantum or len(self.feature_names_in),
            "transformed_feature_names": self.get_transformed_feature_names(),
            "dim_reduction_method": self.config.dim_reduction_method if self.dim_reducer is not None else "none",
        }

        if isinstance(self.dim_reducer, PCA):
            meta["pca_explained_variance_ratio"] = [float(v) for v in self.dim_reducer.explained_variance_ratio_]
            meta["pca_cumulative_variance"] = float(np.sum(self.dim_reducer.explained_variance_ratio_))
        elif isinstance(self.dim_reducer, SelectKBest):
            if hasattr(self.dim_reducer, "scores_") and self.feature_names_in:
                scores_dict = {}
                for idx, name in enumerate(self.feature_names_in):
                    scores_dict[name] = float(self.dim_reducer.scores_[idx])
                meta["feature_importance_scores"] = scores_dict
                selected_indices = self.dim_reducer.get_support(indices=True)
                meta["selected_feature_indices"] = [int(i) for i in selected_indices]

        return meta

