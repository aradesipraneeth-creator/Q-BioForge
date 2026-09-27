"""Quantum experiment orchestration, training, evaluation, and artifact persistence engine for Q-BioForge."""

import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.data.loader import load_biomedical_dataset, BiomedicalDataset
from app.data.preprocessor import PreprocessorConfig, TabularPreprocessor
from app.quantum.devices import get_system_quantum_environment
from app.quantum.vqc import VariationalQuantumClassifier
from app.quantum.qsvm import QuantumSupportVectorClassifier
from app.quantum.metrics import calculate_quantum_metrics, QuantumClassificationMetrics


def get_next_experiment_id(results_dir: str) -> str:
    """Determine the next sequential QB-XXXXX experiment ID without overwriting existing runs."""
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


def train_and_evaluate_quantum_model(
    model_type: str = "vqc",
    dataset_path: Optional[str] = None,
    target_col: str = "target",
    num_qubits: int = 4,
    target_features: int = 4,
    dim_reduction_method: str = "pca",
    encoding_scheme: str = "angle",
    encoding_rotation: str = "Y",
    angle_range: Tuple[float, float] = (0.0, np.pi),
    circuit_depth: int = 2,
    ansatz: str = "hardware_efficient",
    optimizer_name: str = "adam",
    learning_rate: float = 0.05,
    epochs: int = 25,
    batch_size: int = 32,
    random_seed: int = 42,
    noise_model: str = "ideal",
    device_name: Optional[str] = None,
    results_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete zero-leakage training, validation, test evaluation and artifact persistence for a Quantum Model."""
    
    # 1. Resolve paths
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if dataset_path is None:
        dataset_path = os.path.join(backend_root, "datasets", "breast_cancer_wisconsin.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Biomedical dataset missing at: {dataset_path}")

    if results_base_dir is None:
        results_base_dir = os.path.join(backend_root, "results")
    os.makedirs(results_base_dir, exist_ok=True)

    # 2. Inspect Quantum Environment
    env = get_system_quantum_environment()

    # 3. Load dataset with deterministic 70/15/15 stratified partitioning
    ds: BiomedicalDataset = load_biomedical_dataset(
        filepath=dataset_path,
        target_col=target_col,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )

    # 4. Zero-Leakage Preprocessing & Dimensionality Reduction
    prep_cfg = PreprocessorConfig(
        impute_strategy="median",
        scaling_method="quantum_angle",
        target_feature_range=angle_range,
        target_dim_for_quantum=target_features,
        dim_reduction_method=dim_reduction_method,
    )
    preprocessor = TabularPreprocessor(config=prep_cfg)

    # Fit strictly on train split
    x_train_proc = preprocessor.fit_transform(ds.x_train, ds.y_train)
    # Transform val and test using training parameters
    x_val_proc = preprocessor.transform(ds.x_val)
    x_test_proc = preprocessor.transform(ds.x_test)

    y_train = ds.y_train.to_numpy(dtype=int)
    y_val = ds.y_val.to_numpy(dtype=int)
    y_test = ds.y_test.to_numpy(dtype=int)

    transformed_feature_names = preprocessor.get_transformed_feature_names()

    # 5. Instantiate Quantum Classifier
    model_type_clean = model_type.lower().strip()
    if model_type_clean == "vqc":
        model = VariationalQuantumClassifier(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            encoding_rotation=encoding_rotation,
            ansatz=ansatz,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            device_name=device_name,
        )
    elif model_type_clean == "qsvm":
        model = QuantumSupportVectorClassifier(
            num_qubits=num_qubits,
            encoding_rotation=encoding_rotation,
            random_seed=random_seed,
            device_name=device_name,
        )
    else:
        raise ValueError(f"Unknown quantum model type '{model_type}'. Choose from 'vqc' or 'qsvm'.")

    # 6. Fit Model on X_train / y_train with timer
    train_start = time.perf_counter()
    if model_type_clean == "vqc":
        model.fit(x_train_proc, y_train, x_val=x_val_proc, y_val=y_val)
    else:
        model.fit(x_train_proc, y_train)
    training_time_seconds = float(round(time.perf_counter() - train_start, 6))

    # Evaluate Train Split
    y_train_pred = model.predict(x_train_proc)
    y_train_prob = model.predict_proba(x_train_proc)
    train_metrics = calculate_quantum_metrics(y_train, y_train_pred, y_train_prob)

    # Evaluate Validation Split
    y_val_pred = model.predict(x_val_proc)
    y_val_prob = model.predict_proba(x_val_proc)
    val_metrics = calculate_quantum_metrics(y_val, y_val_pred, y_val_prob)

    # 7. Final Test Evaluation (computed strictly once on untouched test split)
    inf_start = time.perf_counter()
    y_test_pred = model.predict(x_test_proc)
    y_test_prob = model.predict_proba(x_test_proc)
    inference_time_seconds = float(round(time.perf_counter() - inf_start, 6))
    test_metrics = calculate_quantum_metrics(y_test, y_test_pred, y_test_prob)

    # 8. Experiment Storage
    experiment_id = get_next_experiment_id(results_base_dir)
    exp_dir = os.path.join(results_base_dir, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat()

    # Save quantum parameters in .npz
    npz_path = os.path.join(exp_dir, "quantum_parameters.npz")
    model.save_checkpoint(npz_path)

    # config.json
    config_dict = {
        "experiment_id": experiment_id,
        "dataset": ds.summary.name,
        "model_type": model_type_clean,
        "model_category": "quantum",
        "framework": "pennylane",
        "qubit_count": num_qubits,
        "feature_count_raw": int(ds.summary.total_features),
        "feature_reduction": dim_reduction_method,
        "reduced_features_count": target_features,
        "selected_features": transformed_feature_names,
        "original_features": ds.summary.feature_names,
        "encoding": encoding_scheme,
        "encoding_rotation": encoding_rotation,
        "angle_range": list(angle_range),
        "circuit_depth": circuit_depth,
        "ansatz": ansatz,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "noise_model": noise_model,
        "device": env["recommended_device"],
        "compute_backend": env["compute_backend"],
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "timestamp": timestamp_iso,
        "status": "completed",
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # metrics.json
    metrics_dict = {
        "experiment_id": experiment_id,
        "model": model_type_clean,
        "model_category": "quantum",
        "dataset": ds.summary.name,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "train_metrics": train_metrics.model_dump(),
        "validation_metrics": val_metrics.model_dump(),
        "test_metrics": test_metrics.model_dump(),
        "evaluation_protocol": "Stratified 70/15/15, fitted strictly on train split, test evaluated once.",
        "scientific_notes": test_metrics.scientific_notes,
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # predictions.json
    predictions_dict = {
        "experiment_id": experiment_id,
        "model": model_type_clean,
        "validation": {
            "y_true": [int(v) for v in y_val],
            "y_pred": [int(v) for v in y_val_pred],
            "y_prob": [float(p[1]) for p in y_val_prob],
        },
        "test": {
            "y_true": [int(v) for v in y_test],
            "y_pred": [int(v) for v in y_test_pred],
            "y_prob": [float(p[1]) for p in y_test_prob],
        }
    }
    with open(os.path.join(exp_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions_dict, f, indent=2)

    # model_metadata.json
    metadata_dict = {
        "experiment_id": experiment_id,
        "model_name": model_type_clean,
        "framework": "pennylane",
        "checkpoint_file": "quantum_parameters.npz",
        "feature_names": transformed_feature_names,
        "feature_count": target_features,
        "qubit_count": num_qubits,
        "circuit_depth": circuit_depth,
        "is_fitted": True,
        "timestamp": timestamp_iso,
    }
    with open(os.path.join(exp_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)

    return {
        "experiment_id": experiment_id,
        "experiment_dir": exp_dir,
        "config": config_dict,
        "metrics": metrics_dict,
        "checkpoint_file": npz_path,
        "environment": env,
    }


def list_all_quantum_results(results_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan results directory and list all executed quantum experiments."""
    if results_base_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_base_dir = os.path.join(backend_root, "results")

    if not os.path.exists(results_base_dir):
        return []

    results = []
    pattern = re.compile(r"^QB-(\d{5})$")

    for entry in sorted(os.listdir(results_base_dir)):
        if pattern.match(entry):
            exp_dir = os.path.join(results_base_dir, entry)
            cfg_path = os.path.join(exp_dir, "config.json")
            met_path = os.path.join(exp_dir, "metrics.json")
            if os.path.exists(cfg_path) and os.path.exists(met_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    with open(met_path, "r", encoding="utf-8") as f:
                        met = json.load(f)
                    if cfg.get("model_category") == "quantum" or cfg.get("model_type") in ["vqc", "qsvm"]:
                        results.append({
                            "experiment_id": entry,
                            "model": cfg.get("model_type", "vqc"),
                            "qubits": cfg.get("qubit_count", 4),
                            "encoding": cfg.get("encoding", "angle"),
                            "circuit_depth": cfg.get("circuit_depth", 2),
                            "test_accuracy": met.get("test_metrics", {}).get("accuracy"),
                            "test_f1": met.get("test_metrics", {}).get("f1_score"),
                            "training_time": met.get("training_time_seconds"),
                            "timestamp": cfg.get("timestamp"),
                        })
                except Exception:
                    continue
    return results


def get_quantum_result(experiment_id: str, results_base_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve complete artifacts for a specific quantum experiment ID."""
    if results_base_dir is None:
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_base_dir = os.path.join(backend_root, "results")

    exp_dir = os.path.join(results_base_dir, experiment_id)
    if not os.path.isdir(exp_dir):
        return None

    cfg_path = os.path.join(exp_dir, "config.json")
    met_path = os.path.join(exp_dir, "metrics.json")
    pred_path = os.path.join(exp_dir, "predictions.json")
    meta_path = os.path.join(exp_dir, "model_metadata.json")
    npz_path = os.path.join(exp_dir, "quantum_parameters.npz")

    res = {"experiment_id": experiment_id}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            res["config"] = json.load(f)
    if os.path.exists(met_path):
        with open(met_path, "r", encoding="utf-8") as f:
            res["metrics"] = json.load(f)
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            res["predictions"] = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            res["metadata"] = json.load(f)
    res["has_checkpoint"] = os.path.exists(npz_path)

    return res
