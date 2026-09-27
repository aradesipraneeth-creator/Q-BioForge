"""Quantum Support Vector Machine (QSVM) and Quantum Kernel estimation for Q-BioForge.

Features:
- Quantum Kernel Matrix computation via PennyLane projective state-overlap circuits: K(x, x') = |<0| U^dag(x') U(x) |0>|^2
- Dual-support for scikit-learn precomputed kernel SVC
- Checkpoint persistence in .npz format
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pennylane as qml
from sklearn.svm import SVC

from app.quantum.base import QuantumClassifier, QuantumFramework, EncodingScheme
from app.quantum.devices import create_quantum_device


class QuantumKernelEstimator:
    """Computes quantum fidelity kernel matrices using PennyLane quantum state overlaps."""

    def __init__(
        self,
        num_qubits: int = 4,
        encoding_rotation: str = "Y",
        device_name: Optional[str] = None,
        shots: Optional[int] = None,
    ):
        self.num_qubits = num_qubits
        self.encoding_rotation = encoding_rotation.upper()
        self.dev = create_quantum_device(wires=self.num_qubits, device_name=device_name, shots=shots)
        self._init_kernel_qnode()

    def _init_kernel_qnode(self) -> None:
        """Define the state overlap fidelity QNode."""
        n_q = self.num_qubits
        enc_rot = self.encoding_rotation

        @qml.qnode(self.dev, interface="autograd")
        def _kernel_circuit(x1, x2):
            # Apply feature map U(x1)
            for i in range(min(len(x1), n_q)):
                if enc_rot == "X":
                    qml.RX(x1[i], wires=i)
                elif enc_rot == "Y":
                    qml.RY(x1[i], wires=i)
                elif enc_rot == "Z":
                    qml.RZ(x1[i], wires=i)

            # Apply adjoint feature map U^dagger(x2)
            for i in range(min(len(x2), n_q) - 1, -1, -1):
                if enc_rot == "X":
                    qml.RX(-x2[i], wires=i)
                elif enc_rot == "Y":
                    qml.RY(-x2[i], wires=i)
                elif enc_rot == "Z":
                    qml.RZ(-x2[i], wires=i)

            # Return probability of projection onto |0...0> state
            return qml.probs(wires=range(n_q))

        self._kernel_qnode = _kernel_circuit

    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute individual fidelity |<phi(x1)|phi(x2)>|^2."""
        probs = self._kernel_qnode(x1, x2)
        # Probability of state |00...0> (index 0)
        return float(probs[0])

    def compute_kernel_matrix(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Compute Gram matrix between dataset A and dataset B."""
        n_a = len(x_a)
        n_b = len(x_b)
        kernel_mat = np.zeros((n_a, n_b), dtype=np.float64)

        is_symmetric = (x_a is x_b) or np.array_equal(x_a, x_b)

        for i in range(n_a):
            j_start = i if is_symmetric else 0
            for j in range(j_start, n_b):
                k_val = self.compute_kernel_element(x_a[i], x_b[j])
                kernel_mat[i, j] = k_val
                if is_symmetric:
                    kernel_mat[j, i] = k_val

        return kernel_mat


class QuantumSupportVectorClassifier(QuantumClassifier):
    """QSVM classifier combining PennyLane quantum kernel with support vector optimization."""

    def __init__(
        self,
        num_qubits: int = 4,
        encoding_rotation: str = "Y",
        c_param: float = 1.0,
        random_seed: int = 42,
        device_name: Optional[str] = None,
        shots: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            num_qubits=num_qubits,
            circuit_depth=1,
            encoding=EncodingScheme.ANGLE,
            framework=QuantumFramework.PENNYLANE,
            config=config or {},
        )
        self.encoding_rotation = encoding_rotation.upper()
        self.c_param = c_param
        self.random_seed = random_seed
        self.kernel_estimator = QuantumKernelEstimator(
            num_qubits=num_qubits,
            encoding_rotation=encoding_rotation,
            device_name=device_name,
            shots=shots,
        )
        self.svc = SVC(C=c_param, kernel="precomputed", probability=True, random_state=random_seed)
        self.x_train_ref: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "QuantumSupportVectorClassifier":
        """Fit QSVM on training data by computing quantum kernel Gram matrix."""
        self.x_train_ref = np.asarray(x_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=int)

        gram_train = self.kernel_estimator.compute_kernel_matrix(self.x_train_ref, self.x_train_ref)
        self.svc.fit(gram_train, y_arr)
        self.is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels for sample x."""
        if not self.is_fitted or self.x_train_ref is None:
            raise RuntimeError("QSVM must be fitted before predict.")
        x_arr = np.asarray(x, dtype=np.float64)
        gram_test = self.kernel_estimator.compute_kernel_matrix(x_arr, self.x_train_ref)
        return self.svc.predict(gram_test)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for sample x."""
        if not self.is_fitted or self.x_train_ref is None:
            raise RuntimeError("QSVM must be fitted before predict_proba.")
        x_arr = np.asarray(x, dtype=np.float64)
        gram_test = self.kernel_estimator.compute_kernel_matrix(x_arr, self.x_train_ref)
        return self.svc.predict_proba(gram_test)

    def save_checkpoint(self, filepath: str) -> str:
        """Save QSVM support vectors, dual coefficients, and intercept in .npz format."""
        if not self.is_fitted or self.x_train_ref is None:
            raise RuntimeError("Cannot save unfitted QSVM.")

        if not filepath.endswith(".npz"):
            filepath = f"{os.path.splitext(filepath)[0]}.npz"

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            x_train_ref=self.x_train_ref,
            dual_coef=self.svc.dual_coef_,
            intercept=self.svc.intercept_,
            support_indices=self.svc.support_,
            num_qubits=np.array([self.num_qubits]),
            c_param=np.array([self.c_param]),
            encoding_rotation=np.array([self.encoding_rotation]),
            random_seed=np.array([self.random_seed]),
        )
        return filepath

    def export_parameters(self) -> Dict[str, Any]:
        """Export model parameters and support vector statistics."""
        return {
            "model_type": "qsvm",
            "framework": "pennylane",
            "num_qubits": self.num_qubits,
            "c_param": self.c_param,
            "support_vector_count": int(len(self.svc.support_)) if self.is_fitted else 0,
            "encoding_rotation": self.encoding_rotation,
            "is_fitted": self.is_fitted,
        }
