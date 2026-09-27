"""Parameterized quantum circuits (ansätze) for Variational Quantum Classifiers in Q-BioForge.

Provides modular variational layers:
- Hardware-Efficient Ansatz (Rotations + CNOT Entanglement)
- Strongly Entangling Layers
- Basic Entangling Layers
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pennylane as qml


def hardware_efficient_layer(
    weights: np.ndarray,
    wires: Sequence[int],
    entanglement: str = "ring",
) -> None:
    """Apply a single hardware-efficient variational layer: Rotations + CNOT entanglers.
    
    weights shape: (len(wires), 3) -> Rotations (Rot(phi, theta, omega) on each wire)
    """
    n_wires = len(wires)
    for i, wire in enumerate(wires):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)

    # Entanglement pattern
    if n_wires > 1:
        for i in range(n_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        if entanglement == "ring" and n_wires > 2:
            qml.CNOT(wires=[wires[n_wires - 1], wires[0]])


def get_vqc_parameter_shape(
    num_qubits: int,
    circuit_depth: int,
    ansatz: str = "hardware_efficient",
) -> Tuple[int, ...]:
    """Calculate the expected parameter tensor shape for a given ansatz configuration."""
    if ansatz in ["hardware_efficient", "strongly_entangling"]:
        return (circuit_depth, num_qubits, 3)
    elif ansatz == "basic_entangling":
        return (circuit_depth, num_qubits)
    else:
        return (circuit_depth, num_qubits, 3)


def build_vqc_circuit(
    features: Any,
    weights: Any,
    num_qubits: int,
    circuit_depth: int,
    encoding_scheme: str = "angle",
    encoding_rotation: str = "Y",
    ansatz: str = "hardware_efficient",
    entanglement: str = "ring",
    measurement_wire: int = 0,
):
    """PennyLane QNode circuit definition for VQC execution supporting autograd tracers, multiple encodings and batching."""
    wires = list(range(num_qubits))
    is_1d = (qml.math.ndim(features) == 1)

    # 1. Feature Encoding (if not data-reuploading, encode at the beginning)
    if encoding_scheme == "amplitude":
        # Amplitude Encoding: state preparation
        f_raw = np.asarray(features, dtype=np.float64)
        if f_raw.ndim == 1:
            padded = np.zeros(2 ** num_qubits, dtype=np.float64)
            padded[:min(len(f_raw), len(padded))] = f_raw[:min(len(f_raw), len(padded))]
            norm = np.linalg.norm(padded)
            padded = padded / (norm if norm > 0 else 1.0)
            qml.StatePrep(padded, wires=wires)
    elif encoding_scheme != "data_reuploading":
        # Standard Angle Encoding
        if is_1d:
            feat_len = len(features)
            for i in range(min(feat_len, num_qubits)):
                if encoding_rotation.upper() == "X":
                    qml.RX(features[i], wires=i)
                elif encoding_rotation.upper() == "Y":
                    qml.RY(features[i], wires=i)
                elif encoding_rotation.upper() == "Z":
                    qml.RZ(features[i], wires=i)
        else:
            n_cols = qml.math.shape(features)[1]
            for i in range(min(n_cols, num_qubits)):
                if encoding_rotation.upper() == "X":
                    qml.RX(features[:, i], wires=i)
                elif encoding_rotation.upper() == "Y":
                    qml.RY(features[:, i], wires=i)
                elif encoding_rotation.upper() == "Z":
                    qml.RZ(features[:, i], wires=i)

    # 2. Variational Layers
    if ansatz == "strongly_entangling":
        qml.StronglyEntanglingLayers(weights, wires=wires)
    elif ansatz == "basic_entangling":
        qml.BasicEntanglingLayers(weights, wires=wires)
    else:
        # Hardware efficient layers (with optional Data Re-uploading per layer)
        for layer in range(circuit_depth):
            if encoding_scheme == "data_reuploading":
                if is_1d:
                    for i in range(min(len(features), num_qubits)):
                        qml.RY(features[i], wires=i)
                else:
                    for i in range(min(qml.math.shape(features)[1], num_qubits)):
                        qml.RY(features[:, i], wires=i)
            hardware_efficient_layer(weights[layer], wires=wires, entanglement=entanglement)

    # 3. Measurement: PauliZ expectation value on measurement_wire
    return qml.expval(qml.PauliZ(measurement_wire))
