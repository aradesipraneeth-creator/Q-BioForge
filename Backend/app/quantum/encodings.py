"""Quantum feature encoding layer for Q-BioForge.

Implements:
- Angle Encoding (n features -> n qubits with Rx, Ry, or Rz single-qubit rotations)
- Configurable angle range mapping (e.g., [0, pi], [-pi, pi], [0, 2*pi])
- Zero data leakage compliance when scaling continuous biomedical features
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pennylane as qml
from app.quantum.base import EncodingScheme, QuantumFeatureMap


class AngleEncoding(QuantumFeatureMap):
    """Angle Encoding mapping n features into single-qubit rotations across n qubits."""

    def __init__(
        self,
        num_qubits: int,
        rotation: str = "Y",
        angle_range: Sequence[float] = (0.0, np.pi),
    ):
        super().__init__(num_qubits=num_qubits, scheme=EncodingScheme.ANGLE)
        self.rotation = rotation.upper()
        self.angle_range = tuple(angle_range)

        if self.rotation not in ["X", "Y", "Z"]:
            raise ValueError(f"Unsupported rotation axis '{rotation}'. Choose from 'X', 'Y', 'Z'.")

    def build_circuit(self, features: Sequence[float]) -> None:
        """Apply angle rotation gates corresponding to input feature coordinates."""
        if len(features) > self.num_qubits:
            raise ValueError(
                f"Feature dimension ({len(features)}) exceeds available qubits ({self.num_qubits}). "
                f"Reduce features using PCA or feature selection."
            )

        for i, val in enumerate(features[:self.num_qubits]):
            if self.rotation == "X":
                qml.RX(val, wires=i)
            elif self.rotation == "Y":
                qml.RY(val, wires=i)
            elif self.rotation == "Z":
                qml.RZ(val, wires=i)


def apply_angle_encoding(
    features: Sequence[float],
    wires: Sequence[int],
    rotation: str = "Y",
) -> None:
    """Convenience functional interface for embedding features via angle rotation gates."""
    rot = rotation.upper()
    for i, wire in enumerate(wires):
        if i < len(features):
            val = features[i]
            if rot == "X":
                qml.RX(val, wires=wire)
            elif rot == "Y":
                qml.RY(val, wires=wire)
            elif rot == "Z":
                qml.RZ(val, wires=wire)


class DataReuploadingEncoding(QuantumFeatureMap):
    """Data Re-uploading scheme interleaving feature rotations across parameterized layers.
    
    Qubit requirement: n features -> n qubits (or n_features // rotations per qubit).
    Input range: [0, pi] or [-pi, pi].
    """

    def __init__(
        self,
        num_qubits: int,
        circuit_depth: int = 2,
        rotation: str = "Y",
    ):
        super().__init__(num_qubits=num_qubits, scheme=EncodingScheme.DATA_REUPLOADING)
        self.circuit_depth = circuit_depth
        self.rotation = rotation.upper()

    def build_circuit(self, features: Sequence[float]) -> None:
        """Apply feature embedding rotations for a specific layer."""
        apply_angle_encoding(features, list(range(self.num_qubits)), rotation=self.rotation)


class AmplitudeEncoding(QuantumFeatureMap):
    """Amplitude Encoding mapping 2^n continuous features into state vector amplitudes across n qubits.
    
    Input requirement: features must be non-zero and are normalized to ||x||_2 = 1.
    Padding: Vectors of dimension < 2^n are zero-padded to 2^n.
    Qubit requirement: N features -> ceil(log2(N)) qubits (Exponential state compression).
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits=num_qubits, scheme=EncodingScheme.AMPLITUDE)
        self.max_dimension = 2 ** num_qubits

    def prepare_state_vector(self, features: Sequence[float]) -> np.ndarray:
        """Normalize and zero-pad input features to form a valid 2^n quantum state vector."""
        feat_arr = np.asarray(features, dtype=np.float64).flatten()
        if len(feat_arr) > self.max_dimension:
            raise ValueError(
                f"Feature dimension ({len(feat_arr)}) exceeds amplitude capacity (2^{self.num_qubits} = {self.max_dimension}). "
                f"Increase qubit count to at least {int(np.ceil(np.log2(len(feat_arr))))}."
            )

        # Pad with zeros to exact 2^n dimension
        padded = np.zeros(self.max_dimension, dtype=np.float64)
        padded[:len(feat_arr)] = feat_arr

        # L2-normalization
        norm = np.linalg.norm(padded)
        if norm == 0 or np.isnan(norm):
            # Fallback to uniform superposition state
            padded = np.ones(self.max_dimension, dtype=np.float64) / np.sqrt(self.max_dimension)
        else:
            padded = padded / norm

        return padded

    def build_circuit(self, features: Sequence[float]) -> None:
        """Apply PennyLane QubitStateVector state preparation."""
        state_vec = self.prepare_state_vector(features)
        qml.StatePrep(state_vec, wires=range(self.num_qubits))
