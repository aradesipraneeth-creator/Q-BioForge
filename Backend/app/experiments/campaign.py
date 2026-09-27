"""Experiment Campaign Orchestrator and Controlled Execution Scheduler for Q-BioForge."""

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.classical.trainer import train_and_evaluate_classical_model
from app.quantum.trainer import train_and_evaluate_quantum_model
from app.experiments.database import insert_or_update_experiment, log_audit_event


MAX_PARALLEL_EXPERIMENTS = 2
LOCAL_CPU_MAX_QUBITS = 16  # State-vector simulation limit for local CPU prototype


class CampaignRunner:
    """Orchestrates multi-experiment campaigns across classical, quantum, and noise dimensions."""

    def __init__(self, campaign_id: str, name: str, description: Optional[str] = None):
        self.campaign_id = campaign_id
        self.name = name
        self.description = description or ""
        self.experiments_queue: List[Dict[str, Any]] = []
        self.completed_experiments: List[Dict[str, Any]] = []
        self.failed_experiments: List[Dict[str, Any]] = []
        self.resource_limited_experiments: List[Dict[str, Any]] = []
        self.status = "QUEUED"

    def add_experiment(self, exp_config: Dict[str, Any]) -> None:
        """Add an experiment to the campaign execution queue."""
        self.experiments_queue.append(exp_config)

    def _execute_single_experiment(self, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single classical or quantum experiment with resource limit guards."""
        model_type = exp_config.get("model", exp_config.get("model_type", "logistic_regression")).lower()
        qubits = exp_config.get("qubit_count", exp_config.get("target_features", 4))

        # Check resource limits for quantum simulation on local CPU
        if model_type in ["vqc", "qsvm", "hybrid_qml"] and qubits > LOCAL_CPU_MAX_QUBITS:
            limit_res = {
                "experiment_id": f"QB-LIMIT-{int(time.time()*1000)%100000:05d}",
                "campaign_id": self.campaign_id,
                "config": {
                    **exp_config,
                    "status": "RESOURCE_LIMITED",
                    "reason": f"Qubit count ({qubits}) exceeds local CPU simulator threshold ({LOCAL_CPU_MAX_QUBITS}). Requires DGX B200 HPC execution.",
                },
                "metrics": {},
                "status": "RESOURCE_LIMITED",
            }
            log_audit_event(
                action="EXPERIMENT_RESOURCE_LIMITED",
                experiment_id=limit_res["experiment_id"],
                details=limit_res["config"]["reason"],
            )
            return limit_res

        # Classical baseline execution
        if model_type in ["logistic_regression", "random_forest", "svm", "xgboost"]:
            res = train_and_evaluate_classical_model(
                model_type=model_type,
                dataset_path=exp_config.get("dataset_path"),
                random_seed=exp_config.get("random_seed", 42),
                hyperparameters=exp_config.get("hyperparameters"),
            )
            res["campaign_id"] = self.campaign_id
            insert_or_update_experiment(res)
            log_audit_event(
                action="CLASSICAL_EXPERIMENT_COMPLETED",
                experiment_id=res["experiment_id"],
                details=f"Model: {model_type}",
            )
            return res

        # Quantum / Hybrid model execution
        elif model_type in ["vqc", "qsvm"]:
            res = train_and_evaluate_quantum_model(
                model_type=model_type,
                dataset_path=exp_config.get("dataset_path"),
                num_qubits=exp_config.get("num_qubits", 4),
                target_features=exp_config.get("target_features", 4),
                dim_reduction_method=exp_config.get("dim_reduction_method", "pca"),
                encoding_scheme=exp_config.get("encoding_scheme", "angle"),
                encoding_rotation=exp_config.get("encoding_rotation", "Y"),
                circuit_depth=exp_config.get("circuit_depth", 2),
                optimizer_name=exp_config.get("optimizer_name", "adam"),
                learning_rate=exp_config.get("learning_rate", 0.05),
                epochs=exp_config.get("epochs", 25),
                batch_size=exp_config.get("batch_size", 32),
                random_seed=exp_config.get("random_seed", 42),
                noise_model=exp_config.get("noise_model", "ideal"),
            )
            res["campaign_id"] = self.campaign_id
            insert_or_update_experiment(res)
            log_audit_event(
                action="QUANTUM_EXPERIMENT_COMPLETED",
                experiment_id=res["experiment_id"],
                details=f"Model: {model_type}, Qubits: {exp_config.get('num_qubits', 4)}",
            )
            return res
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for campaign execution.")

    def run_campaign(self, max_parallel: int = MAX_PARALLEL_EXPERIMENTS) -> Dict[str, Any]:
        """Execute all queued campaign experiments."""
        self.status = "RUNNING"
        start_time = datetime.now(timezone.utc).isoformat()

        for exp in self.experiments_queue:
            try:
                res = self._execute_single_experiment(exp)
                if res.get("status") == "RESOURCE_LIMITED":
                    self.resource_limited_experiments.append(res)
                else:
                    self.completed_experiments.append(res)
            except Exception as e:
                failed_entry = {
                    "config": exp,
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.failed_experiments.append(failed_entry)
                log_audit_event(
                    action="EXPERIMENT_FAILED",
                    details=f"Error: {str(e)}",
                )

        self.status = "COMPLETED"
        end_time = datetime.now(timezone.utc).isoformat()

        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "status": self.status,
            "total_queued": len(self.experiments_queue),
            "total_completed": len(self.completed_experiments),
            "total_failed": len(self.failed_experiments),
            "total_resource_limited": len(self.resource_limited_experiments),
            "start_time": start_time,
            "end_time": end_time,
            "completed_experiments": [e.get("experiment_id") for e in self.completed_experiments],
        }
