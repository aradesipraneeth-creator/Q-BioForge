"""Biomedical tabular dataset loader and schema validator for Q-BioForge.

Enforces:
- Schema verification & missing value auditing
- Class balance & distribution profiling
- Stratified partitioning into Train (70%), Validation (15%), and Test (15%) splits
- Zero-leakage contract (pure data extraction before pipeline transforms)
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class DatasetSummary(BaseModel):
    """Statistical summary and validation metadata of a loaded biomedical dataset."""
    name: str
    filepath: str
    total_samples: int
    total_features: int
    feature_names: List[str]
    target_name: str
    missing_values_count: int
    missing_by_column: Dict[str, int]
    class_distribution: Dict[str, int]
    class_proportions: Dict[str, float]
    train_samples: int
    val_samples: int
    test_samples: int


class BiomedicalDataset:
    """Container holding raw and split partitions of a biomedical dataset."""

    def __init__(
        self,
        name: str,
        x_raw: pd.DataFrame,
        y_raw: pd.Series,
        x_train: pd.DataFrame,
        x_val: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        summary: DatasetSummary,
    ):
        self.name = name
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.summary = summary


def load_biomedical_dataset(
    filepath: str,
    target_col: str = "target",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> BiomedicalDataset:
    """Load, validate schema, inspect distributions, and split a tabular biomedical dataset.
    
    Args:
        filepath: Path to CSV dataset.
        target_col: Column name of target label.
        train_ratio: Proportion of data for training split.
        val_ratio: Proportion of data for validation split.
        test_ratio: Proportion of data for test split.
        random_seed: Seed for deterministic stratified partitioning.
        
    Returns:
        BiomedicalDataset containing stratified partitions and metadata summary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Biomedical dataset not found at path: {filepath}")

    # 1. Load CSV
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    # 2. Identify and validate target column
    if target_col not in df.columns:
        # Fallback heuristic: check if last column is binary/target
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) == 0:
        raise ValueError("Dataset has no feature columns.")

    x_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # 3. Missing values analysis
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = sum(missing_by_col.values())

    # 4. Class distribution analysis
    value_counts = y_raw.value_counts().to_dict()
    total_count = len(y_raw)
    class_dist = {str(k): int(v) for k, v in value_counts.items()}
    class_props = {str(k): round(float(v) / total_count, 4) for k, v in value_counts.items()}

    # 5. Stratified Train / Validation / Test Splitting
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_raw,
        y_raw,
        test_size=temp_ratio,
        stratify=y_raw,
        random_state=random_seed,
    )

    # Second split: Validation vs Test
    val_rel_ratio = val_ratio / temp_ratio
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1.0 - val_rel_ratio),
        stratify=y_temp,
        random_state=random_seed,
    )

    dataset_name = os.path.splitext(os.path.basename(filepath))[0]

    summary = DatasetSummary(
        name=dataset_name,
        filepath=filepath,
        total_samples=total_count,
        total_features=len(feature_cols),
        feature_names=feature_cols,
        target_name=target_col,
        missing_values_count=total_missing,
        missing_by_column=missing_by_col,
        class_distribution=class_dist,
        class_proportions=class_props,
        train_samples=len(x_train),
        val_samples=len(x_val),
        test_samples=len(x_test),
    )

    return BiomedicalDataset(
        name=dataset_name,
        x_raw=x_raw,
        y_raw=y_raw,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        summary=summary,
    )
