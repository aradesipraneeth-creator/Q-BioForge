"""Model Registry for Classical, Quantum, and Hybrid Biomedical Models in Q-BioForge."""

import os
import json
import joblib
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from app.classical.base import ClassicalModel
from app.classical.trainer import get_model_instance
from app.quantum.vqc import VariationalQuantumClassifier
from app.quantum.qsvm import QuantumSupportVectorClassifier
from app.quantum.hybrid import HybridQuantumClassifier


class ModelRegistry:
    """Central registry for discovering, loading, and predicting with trained models."""

    def __init__(self, results_base_dir: Optional[str] = None):
        if results_base_dir is None:
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_base_dir = os.path.join(backend_root, "results")
        self.results_base_dir = results_base_dir

    def list_models(self) -> List[Dict[str, Any]]:
        """List all verified trained models in the results directory."""
        if not os.path.exists(self.results_base_dir):
            return []

        models = []
        for exp_id in sorted(os.listdir(self.results_base_dir)):
            exp_dir = os.path.join(self.results_base_dir, exp_id)
            if not os.path.isdir(exp_dir):
                continue

            cfg_path = os.path.join(exp_dir, "config.json")
            met_path = os.path.join(exp_dir, "metrics.json")
            meta_path = os.path.join(exp_dir, "model_metadata.json")

            if os.path.exists(cfg_path) and os.path.exists(met_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    with open(met_path, "r", encoding="utf-8") as f:
                        met = json.load(f)

                    model_type = cfg.get("model", cfg.get("model_type", "unknown"))
                    model_cat = cfg.get("model_category", "classical" if model_type in ["logistic_regression", "random_forest", "svm", "xgboost"] else "quantum")

                    test_m = met.get("test_metrics", {})
                    models.append({
                        "model_id": f"MOD-{exp_id}",
                        "experiment_id": exp_id,
                        "model_name": model_type,
                        "model_category": model_cat,
                        "dataset": cfg.get("dataset", "breast_cancer_wisconsin"),
                        "qubit_count": cfg.get("qubit_count"),
                        "circuit_depth": cfg.get("circuit_depth"),
                        "encoding": cfg.get("encoding"),
                        "noise_model": cfg.get("noise_model", "ideal"),
                        "accuracy": test_m.get("accuracy"),
                        "f1_score": test_m.get("f1_score"),
                        "roc_auc": test_m.get("roc_auc"),
                        "sensitivity": test_m.get("sensitivity"),
                        "specificity": test_m.get("specificity"),
                        "training_time_seconds": met.get("training_time_seconds"),
                        "timestamp": cfg.get("timestamp"),
                        "status": cfg.get("status", "completed"),
                    })
                except Exception:
                    continue

        return models

    def load_model(self, model_id_or_exp_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a trained model instance and its configuration."""
        exp_id = model_id_or_exp_id.replace("MOD-", "")
        exp_dir = os.path.join(self.results_base_dir, exp_id)
        if not os.path.isdir(exp_dir):
            # Fallback: search for model by model_name or model_id in registered experiments
            all_models = self.list_models()
            matched = [
                m for m in all_models
                if m.get("model_name", "").lower() == model_id_or_exp_id.lower()
                or m.get("model_id") == model_id_or_exp_id
            ]
            if matched:
                exp_id = matched[0]["experiment_id"]
                exp_dir = os.path.join(self.results_base_dir, exp_id)
            else:
                raise FileNotFoundError(f"Model checkpoint directory not found for: {model_id_or_exp_id}")

        cfg_path = os.path.join(exp_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_type = cfg.get("model", cfg.get("model_type", "")).lower()

        # Classical Joblib model
        joblib_path = os.path.join(exp_dir, "model_checkpoint.joblib")
        if os.path.exists(joblib_path):
            model_wrapper = get_model_instance(model_type, config=cfg.get("hyperparameters", {}))
            model_wrapper.load_checkpoint(joblib_path)
            return model_wrapper, cfg

        # Quantum NPZ model
        npz_path = os.path.join(exp_dir, "quantum_parameters.npz")
        if os.path.exists(npz_path):
            if model_type == "vqc":
                vqc = VariationalQuantumClassifier()
                vqc.load_checkpoint(npz_path)
                return vqc, cfg
            elif model_type == "qsvm":
                qsvm = QuantumSupportVectorClassifier()
                # Load checkpoint
                data = np.load(npz_path, allow_pickle=True)
                qsvm.x_train_ref = data["x_train_ref"]
                qsvm.svc.dual_coef_ = data["dual_coef"]
                qsvm.svc.intercept_ = data["intercept"]
                qsvm.svc.support_ = data["support_indices"]
                qsvm.is_fitted = True
                return qsvm, cfg
            elif model_type == "hybrid_qml":
                hybrid = HybridQuantumClassifier()
                hybrid.load_checkpoint(npz_path)
                return hybrid, cfg

        raise FileNotFoundError(f"No valid .joblib or .npz checkpoint found in {exp_dir}")

    def predict(self, model_id: str, input_features: np.ndarray) -> Dict[str, Any]:
        """Execute inference with registered model on input feature vector."""
        model, cfg = self.load_model(model_id)
        features_arr = np.asarray(input_features, dtype=np.float64)
        if features_arr.ndim == 1:
            features_arr = features_arr.reshape(1, -1)

        preds = model.predict(features_arr)
        probs = model.predict_proba(features_arr)

        return {
            "model_id": model_id,
            "prediction": int(preds[0]),
            "predicted_class": "Malignant" if preds[0] == 1 else "Benign",
            "probabilities": {
                "class_0_benign": float(probs[0, 0]),
                "class_1_malignant": float(probs[0, 1]),
            },
            "confidence": float(max(probs[0, 0], probs[0, 1])),
            "model_type": cfg.get("model", cfg.get("model_type")),
        }
