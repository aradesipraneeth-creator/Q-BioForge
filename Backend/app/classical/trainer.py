"""Classical baseline training, evaluation, and experiment persistence engine for Q-BioForge."""

import os
import json
import time
import glob
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import joblib

from app.data.loader import load_biomedical_dataset, BiomedicalDataset
from app.data.preprocessor import PreprocessorConfig, TabularPreprocessor
from app.classical.base import ClassicalModel
from app.classical.logistic_regression import LogisticRegressionModel
from app.classical.random_forest import RandomForestModel
from app.classical.svm import SVMModel
from app.classical.xgboost_model import XGBoostModel
from app.classical.metrics import calculate_metrics, ClassificationMetrics, BenchmarkRecord


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "xgboost": XGBoostModel,
}


def get_model_instance(model_type: str, config: Optional[Dict[str, Any]] = None) -> ClassicalModel:
    """Instantiate a classical model from registry."""
    key = model_type.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available classical models: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[key]
    return model_cls(config=config)


def get_next_experiment_id(results_dir: str) -> str:
    """Determine the next sequential QB-XXXXX experiment ID without overwriting existing experiments."""
    os.makedirs(results_dir, exist_ok=True)
    existing_dirs = os.listdir(results_dir)
    pattern = re.compile(r"^QB-(\d{5})$")
    max_id = 0
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            num = int(match.group(1))
            if num > max_id:
                max_id = num
    next_id = max_id + 1
    return f"QB-{next_id:05d}"


def train_and_evaluate_classical_model(
    model_type: str,
    dataset_path: Optional[str] = None,
    target_col: str = "target",
    random_seed: int = 42,
    hyperparameters: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Union[PreprocessorConfig, Dict[str, Any]]] = None,
    results_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full zero-leakage training, validation, and final test evaluation for a classical baseline model.
    
    Workflow:
    1. Load dataset with deterministic 70/15/15 stratified partitioning.
    2. Fit preprocessor parameters STRICTLY on the training set (Zero Data Leakage Contract).
    3. Transform validation and test sets using parameters learned from training.
    4. Instantiate real classical model with explicit hyperparameters and deterministic random seed.
    5. Fit model on training split and record training time.
    6. Evaluate on validation split for model/hyperparameter selection.
    7. Final evaluation on test split computed ONLY ONCE on the untouched test partition.
    8. Persist unique experiment artifacts (config.json, metrics.json, predictions.json, model_metadata.json, checkpoint).
    
    Returns:
        Structured experiment result dictionary.
    """
    # 1. Resolve dataset path
    if dataset_path is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(backend_root, "datasets", "breast_cancer_wisconsin.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Biomedical dataset not found at path: {dataset_path}")

    # Resolve results directory
    if results_base_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_base_dir = os.path.join(backend_root, "results")
    os.makedirs(results_base_dir, exist_ok=True)

    # 2. Load dataset
    ds: BiomedicalDataset = load_biomedical_dataset(
        filepath=dataset_path,
        target_col=target_col,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )

    # 3. Setup Preprocessor
    if isinstance(preprocessing_config, PreprocessorConfig):
        prep_cfg = preprocessing_config
    elif isinstance(preprocessing_config, dict):
        prep_cfg = PreprocessorConfig(**preprocessing_config)
    else:
        # Default zero-leakage classical preprocessing: median imputation + standard scaling
        prep_cfg = PreprocessorConfig(
            impute_strategy="median",
            scaling_method="standard",
            target_dim_for_quantum=None,
        )

    preprocessor = TabularPreprocessor(config=prep_cfg)

    # Zero-Leakage Contract: Fit exclusively on training data
    x_train_proc = preprocessor.fit_transform(ds.x_train, ds.y_train)
    # Validation & Test transformed using training-derived statistics
    x_val_proc = preprocessor.transform(ds.x_val)
    x_test_proc = preprocessor.transform(ds.x_test)

    y_train = ds.y_train.to_numpy(dtype=int)
    y_val = ds.y_val.to_numpy(dtype=int)
    y_test = ds.y_test.to_numpy(dtype=int)

    transformed_feature_names = preprocessor.get_transformed_feature_names()
    transformed_feature_count = x_train_proc.shape[1]

    # 4. Model instantiation
    model_cfg = dict(hyperparameters or {})
    model_cfg["random_seed"] = random_seed
    model = get_model_instance(model_type, config=model_cfg)

    # 5. Fit on Train Split and record training duration
    train_start = time.perf_counter()
    model.fit(x_train_proc, y_train, feature_names=transformed_feature_names)
    training_time_seconds = float(round(time.perf_counter() - train_start, 6))

    # Evaluate on Train Split (diagnostic)
    y_train_pred = model.predict(x_train_proc)
    y_train_prob = model.predict_proba(x_train_proc)
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob)

    # 6. Evaluate on Validation Split (for validation / selection)
    y_val_pred = model.predict(x_val_proc)
    y_val_prob = model.predict_proba(x_val_proc)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)

    # 7. Final Test Evaluation (computed ONLY ONCE on untouched test partition)
    inf_start = time.perf_counter()
    y_test_pred = model.predict(x_test_proc)
    y_test_prob = model.predict_proba(x_test_proc)
    inference_time_seconds = float(round(time.perf_counter() - inf_start, 6))
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)

    # 8. Experiment Storage
    experiment_id = get_next_experiment_id(results_base_dir)
    exp_dir = os.path.join(results_base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # config.json
    config_dict = {
        "experiment_id": experiment_id,
        "dataset": ds.summary.name,
        "model": model_type,
        "random_seed": random_seed,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "feature_count": int(ds.summary.total_features),
        "feature_names": ds.summary.feature_names,
        "original_features": ds.summary.feature_names,
        "selected_features": transformed_feature_names,
        "transformed_feature_count": transformed_feature_count,
        "preprocessing_configuration": prep_cfg.model_dump(),
        "hyperparameters": model_cfg,
        "timestamp": timestamp_iso,
        "status": "completed",
    }

    # metrics.json
    metrics_dict = {
        "experiment_id": experiment_id,
        "model": model_type,
        "dataset": ds.summary.name,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "train_metrics": train_metrics.model_dump(),
        "validation_metrics": val_metrics.model_dump(),
        "test_metrics": test_metrics.model_dump(),
        "evaluation_protocol": "Stratified 70/15/15, fitted strictly on train split, test evaluated once.",
    }

    # predictions.json
    predictions_dict = {
        "experiment_id": experiment_id,
        "model": model_type,
        "validation": {
            "y_true": y_val.tolist(),
            "y_pred": [int(p) for p in y_val_pred],
            "y_prob": y_val_prob.tolist() if y_val_prob is not None else [],
        },
        "test": {
            "y_true": y_test.tolist(),
            "y_pred": [int(p) for p in y_test_pred],
            "y_prob": y_test_prob.tolist() if y_test_prob is not None else [],
        },
    }

    # model_metadata.json
    checkpoint_filename = "model_checkpoint.joblib"
    checkpoint_filepath = os.path.join(exp_dir, checkpoint_filename)
    model.save_checkpoint(checkpoint_filepath)

    model_metadata = {
        "experiment_id": experiment_id,
        "model_name": model_type,
        "framework": "xgboost" if model_type == "xgboost" else "scikit-learn",
        "checkpoint_file": checkpoint_filename,
        "feature_names": transformed_feature_names,
        "feature_count": transformed_feature_count,
        "is_fitted": True,
        "timestamp": timestamp_iso,
    }

    # Write JSON files to experiment directory
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(os.path.join(exp_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions_dict, f, indent=2)

    with open(os.path.join(exp_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Return summary dictionary
    return {
        "experiment_id": experiment_id,
        "experiment_dir": exp_dir,
        "config": config_dict,
        "metrics": metrics_dict,
        "model_metadata": model_metadata,
        "benchmark_summary": {
            "experiment_id": experiment_id,
            "model": model_type,
            "dataset": ds.summary.name,
            "accuracy": test_metrics.accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1_score,
            "roc_auc": test_metrics.roc_auc,
            "sensitivity": test_metrics.sensitivity,
            "specificity": test_metrics.specificity,
            "brier_score": test_metrics.brier_score,
            "training_time": training_time_seconds,
            "inference_time": inference_time_seconds,
            "random_seed": random_seed,
            "timestamp": timestamp_iso,
            "status": "completed",
        },
    }


def list_all_experiment_results(results_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve summaries for all stored experiments."""
    if results_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(backend_root, "results")

    if not os.path.exists(results_dir):
        return []

    experiments = []
    pattern = re.compile(r"^QB-\d{5}$")
    for item in sorted(os.listdir(results_dir)):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and pattern.match(item):
            cfg_path = os.path.join(item_path, "config.json")
            met_path = os.path.join(item_path, "metrics.json")
            if os.path.exists(cfg_path) and os.path.exists(met_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    with open(met_path, "r", encoding="utf-8") as f:
                        met = json.load(f)
                    test_m = met.get("test_metrics", {})
                    experiments.append({
                        "experiment_id": cfg.get("experiment_id", item),
                        "model": cfg.get("model"),
                        "dataset": cfg.get("dataset"),
                        "accuracy": test_m.get("accuracy"),
                        "precision": test_m.get("precision"),
                        "recall": test_m.get("recall"),
                        "f1": test_m.get("f1_score"),
                        "roc_auc": test_m.get("roc_auc"),
                        "sensitivity": test_m.get("sensitivity"),
                        "specificity": test_m.get("specificity"),
                        "brier_score": test_m.get("brier_score"),
                        "training_time": met.get("training_time_seconds"),
                        "inference_time": met.get("inference_time_seconds"),
                        "random_seed": cfg.get("random_seed"),
                        "timestamp": cfg.get("timestamp"),
                        "status": cfg.get("status", "completed"),
                    })
                except Exception:
                    continue
    return experiments


def get_experiment_result(experiment_id: str, results_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve complete artifacts for a specific experiment ID."""
    if results_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(backend_root, "results")

    exp_dir = os.path.join(results_dir, experiment_id)
    if not os.path.exists(exp_dir):
        return None

    try:
        with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(exp_dir, "metrics.json"), "r", encoding="utf-8") as f:
            met = json.load(f)
        with open(os.path.join(exp_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        pred_data = {}
        pred_path = os.path.join(exp_dir, "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)

        return {
            "experiment_id": experiment_id,
            "config": cfg,
            "metrics": met,
            "model_metadata": meta,
            "predictions": pred_data,
        }
    except Exception as e:
        return None


def generate_classical_benchmark_table(results_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate a single scientifically neutral classical benchmark table without ranking or quantum comparisons."""
    records = list_all_experiment_results(results_dir)
    return records
