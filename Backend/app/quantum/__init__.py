"""Quantum Machine Learning Engine for Q-BioForge."""

from app.quantum.base import (
    EncodingScheme,
    QuantumFramework,
    QuantumCircuitMetadata,
    QuantumFeatureMap,
    QuantumClassifier,
)
from app.quantum.devices import (
    get_system_quantum_environment,
    create_quantum_device,
)
from app.quantum.encodings import (
    AngleEncoding,
    apply_angle_encoding,
)
from app.quantum.circuits import (
    build_vqc_circuit,
    hardware_efficient_layer,
    get_vqc_parameter_shape,
)
from app.quantum.vqc import VariationalQuantumClassifier
from app.quantum.qsvm import QuantumSupportVectorClassifier, QuantumKernelEstimator
from app.quantum.metrics import (
    QuantumClassificationMetrics,
    calculate_quantum_metrics,
)
from app.quantum.trainer import (
    train_and_evaluate_quantum_model,
    list_all_quantum_results,
    get_quantum_result,
)

__all__ = [
    "EncodingScheme",
    "QuantumFramework",
    "QuantumCircuitMetadata",
    "QuantumFeatureMap",
    "QuantumClassifier",
    "get_system_quantum_environment",
    "create_quantum_device",
    "AngleEncoding",
    "apply_angle_encoding",
    "build_vqc_circuit",
    "hardware_efficient_layer",
    "get_vqc_parameter_shape",
    "VariationalQuantumClassifier",
    "QuantumSupportVectorClassifier",
    "QuantumKernelEstimator",
    "QuantumClassificationMetrics",
    "calculate_quantum_metrics",
    "train_and_evaluate_quantum_model",
    "list_all_quantum_results",
    "get_quantum_result",
]
