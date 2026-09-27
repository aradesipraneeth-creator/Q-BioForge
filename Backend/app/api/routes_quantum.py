"""FastAPI REST API routes for Quantum Lab and Quantum Experiment Orchestration."""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.quantum.devices import get_system_quantum_environment
from app.quantum.compute import get_active_compute_backend
from app.quantum.trainer import (
    train_and_evaluate_quantum_model,
    list_all_quantum_results,
    get_quantum_result,
)
from app.experiments.database import log_audit_event

router = APIRouter(prefix="/quantum", tags=["Quantum Lab"])


class QuantumTrainRequest(BaseModel):
    """Payload for triggering a real quantum model training experiment."""
    model_type: str = Field("vqc", description="Quantum model architecture: 'vqc' or 'qsvm'")
    target_features: int = Field(4, ge=2, le=30, description="Reduced feature count for quantum encoding")
    dim_reduction_method: str = Field("pca", description="Feature reduction method: 'pca' or 'select_k_best'")
    num_qubits: int = Field(4, ge=2, le=30, description="Qubit count")
    circuit_depth: int = Field(2, ge=1, le=10, description="Variational circuit depth")
    encoding_scheme: str = Field("angle", description="Quantum feature encoding scheme")
    encoding_rotation: str = Field("Y", description="Single-qubit rotation gate axis: 'X', 'Y', 'Z'")
    optimizer_name: str = Field("adam", description="Optimizer: 'adam', 'nesterov', 'gradient_descent'")
    learning_rate: float = Field(0.05, ge=0.0001, le=1.0, description="Learning rate")
    epochs: int = Field(25, ge=1, le=100, description="Training epochs")
    batch_size: int = Field(32, ge=1, le=256, description="Mini-batch size")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    noise_model: str = Field("ideal", description="Noise model: 'ideal' or noise presets")


@router.get("/devices", response_model=Dict[str, Any])
def get_quantum_devices():
    """Programmatically detect and return available quantum simulation devices and hardware context."""
    env = get_system_quantum_environment()
    backend = get_active_compute_backend()
    return {
        **env,
        "compute_backend_info": backend.get_status_info(),
        "public_limits": {
            "max_public_qubits": settings.MAX_PUBLIC_QUBITS,
            "max_public_depth": settings.MAX_PUBLIC_DEPTH,
            "max_public_experiments": settings.MAX_PUBLIC_EXPERIMENTS,
        },
    }


@router.get("/results", response_model=List[Dict[str, Any]])
def get_quantum_results():
    """Retrieve summary listing of all executed quantum experiments."""
    return list_all_quantum_results()


@router.get("/results/{experiment_id}", response_model=Dict[str, Any])
def get_quantum_experiment_detail(experiment_id: str):
    """Retrieve full configuration, test metrics, predictions, and metadata for a specific quantum experiment."""
    res = get_quantum_result(experiment_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Quantum experiment '{experiment_id}' not found.")
    return res


@router.post("/train", response_model=Dict[str, Any])
def trigger_quantum_training(request: QuantumTrainRequest):
    """Trigger a real execution of a Quantum Classifier (VQC or QSVM) with public resource safety guards."""
    
    # Check public environment resource limits
    if settings.COMPUTE_BACKEND != "DGX":
        if request.num_qubits > settings.MAX_PUBLIC_QUBITS or request.circuit_depth > settings.MAX_PUBLIC_DEPTH:
            limit_id = f"QB-LIMIT-{int(time.time() * 1000) % 100000:05d}"
            reason = (
                f"Requested configuration ({request.num_qubits} qubits, depth {request.circuit_depth}) exceeds "
                f"public demo runtime limits (max {settings.MAX_PUBLIC_QUBITS} qubits, depth {settings.MAX_PUBLIC_DEPTH}). "
                f"Large-scale circuits are reserved for college NVIDIA DGX B200 HPC campaign execution."
            )
            log_audit_event(
                action="PUBLIC_RESOURCE_LIMITED",
                experiment_id=limit_id,
                details=reason,
            )
            return {
                "experiment_id": limit_id,
                "status": "RESOURCE_LIMITED",
                "message": reason,
                "config": request.model_dump(),
                "metrics": {},
            }

    try:
        result = train_and_evaluate_quantum_model(
            model_type=request.model_type,
            num_qubits=request.num_qubits,
            target_features=request.target_features,
            dim_reduction_method=request.dim_reduction_method,
            encoding_scheme=request.encoding_scheme,
            encoding_rotation=request.encoding_rotation,
            circuit_depth=request.circuit_depth,
            optimizer_name=request.optimizer_name,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            batch_size=request.batch_size,
            random_seed=request.random_seed,
            noise_model=request.noise_model,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantum model training failed: {str(e)}")
