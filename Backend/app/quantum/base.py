"""Base interfaces for Quantum Machine Learning components in Q-BioForge.

Provides hardware-agnostic and simulator-agnostic abstractions supporting:
- Feature encoding (Angle, Amplitude, Data Re-uploading)
- Variational Quantum Classifiers (VQC)
- Quantum Support Vector Machines (QSVM / Quantum Kernels)
- Quantum Neural Networks (QNN) & Hybrid Quantum-Classical models
- Backend-configurable simulation (Local development or NVIDIA DGX B200 HPC)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EncodingScheme(str, Enum):
    ANGLE = "angle"
    AMPLITUDE = "amplitude"
    DATA_REUPLOADING = "data_reuploading"
    CUSTOM = "custom"


class QuantumFramework(str, Enum):
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"


class QuantumCircuitMetadata(BaseModel):
    """Structural metadata for a quantum circuit."""
    qubit_count: int = Field(..., ge=1, description="Number of qubits")
    circuit_depth: int = Field(..., ge=0, description="Circuit depth/layers")
    gate_count: Dict[str, int] = Field(default_factory=dict, description="Count of 1-qubit and 2-qubit gates")
    parameter_count: int = Field(0, ge=0, description="Number of trainable variational parameters")
    framework: QuantumFramework = QuantumFramework.PENNYLANE
    encoding_scheme: EncodingScheme = EncodingScheme.ANGLE


class QuantumBackendProvider(ABC):
    """Interface for dispatching circuit simulation to local or DGX compute."""

    @abstractmethod
    def get_backend_name(self) -> str:
        pass

    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: Optional[int] = None) -> Any:
        pass


class QuantumFeatureMap(ABC):
    """Abstract quantum data encoding interface."""

    def __init__(self, num_qubits: int, scheme: EncodingScheme = EncodingScheme.ANGLE):
        self.num_qubits = num_qubits
        self.scheme = scheme

    @abstractmethod
    def build_circuit(self, features: Any) -> Any:
        """Construct the encoding circuit for input feature vector."""
        pass


class QuantumClassifier(ABC):
    """Base interface for quantum and hybrid classification models."""

    def __init__(
        self,
        num_qubits: int,
        circuit_depth: int = 1,
        encoding: EncodingScheme = EncodingScheme.ANGLE,
        framework: QuantumFramework = QuantumFramework.PENNYLANE,
        config: Optional[Dict[str, Any]] = None
    ):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.encoding = encoding
        self.framework = framework
        self.config = config or {}
        self.is_fitted: bool = False
        self.parameters: Optional[Any] = None

    @abstractmethod
    def fit(self, x_train: Any, y_train: Any) -> "QuantumClassifier":
        """Train variational parameters on preprocessed features."""
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict labels."""
        pass

    @abstractmethod
    def predict_proba(self, x: Any) -> Any:
        """Predict class probability distribution."""
        pass

    @abstractmethod
    def export_parameters(self) -> Dict[str, Any]:
        """Export trained variational parameters and metadata."""
        pass
