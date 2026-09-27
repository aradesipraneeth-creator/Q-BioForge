"""Experiment engine schemas and workflow manager for Q-BioForge.

Enables reproducible experimentation across:
Dataset × Model × Qubit Count × Encoding × Circuit Depth × Noise Model × Optimizer
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentConfig(BaseModel):
    """Configuration for reproducible classical, quantum, or hybrid experiment."""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    dataset: str = Field(..., description="Name of dataset")
    model_type: str = Field(..., description="Classical, quantum, or hybrid model identifier")
    qubit_count: Optional[int] = Field(None, ge=1, description="Qubit count (e.g. 4, 8, 12, 16, 20, 24...)")
    encoding: Optional[str] = Field(None, description="Encoding scheme (e.g., angle, amplitude)")
    circuit_depth: Optional[int] = Field(None, ge=1, description="Variational circuit depth/layers")
    noise_model: Optional[str] = Field("ideal", description="Noise configuration identifier")
    optimizer: Optional[str] = Field(None, description="Optimizer (e.g., Adam, COBYLA, SPSA)")
    seed: int = Field(default=42, description="Random seed for full reproducibility")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model and training hyperparameters")
    compute_backend: str = Field(default="local", description="Target compute backend (local or DGX)")


class ExperimentResult(BaseModel):
    """Execution output and performance metrics of an experiment run."""
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    training_time_seconds: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    reproducibility_seed: int = 42
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelCheckpointMetadata(BaseModel):
    """Metadata specification for saved models in the Model Registry."""
    model_id: str
    experiment_id: str
    dataset: str
    model_type: str
    feature_representation: str
    qubit_count: Optional[int] = None
    encoding: Optional[str] = None
    circuit_depth: Optional[int] = None
    noise_configuration: Optional[Dict[str, Any]] = None
    training_configuration: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoint_format: str = Field(..., description="File format, e.g., .pth for PyTorch, .json/.npz for quantum parameters")
    filepath: str
    created_at: str
