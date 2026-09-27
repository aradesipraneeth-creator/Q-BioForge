"""Variational Quantum Classifier (VQC) implementation using PennyLane for Q-BioForge.

Features:
- Parameterized Quantum Circuit with Angle Encoding and Hardware-Efficient / Strongly Entangling layers
- Configurable qubit count, depth, learning rate, optimizer, epochs, and random seed
- Checkpoint persistence in .npz format with complete metadata
- Support for probability estimation and discrete prediction
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from app.quantum.base import QuantumClassifier, QuantumFramework, EncodingScheme
from app.quantum.devices import create_quantum_device
from app.quantum.circuits import build_vqc_circuit, get_vqc_parameter_shape


class VariationalQuantumClassifier(QuantumClassifier):
    """Real Variational Quantum Classifier (VQC) with PennyLane execution."""

    def __init__(
        self,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        encoding_rotation: str = "Y",
        ansatz: str = "hardware_efficient",
        optimizer_name: str = "adam",
        learning_rate: float = 0.05,
        epochs: int = 30,
        batch_size: int = 32,
        random_seed: int = 42,
        device_name: Optional[str] = None,
        shots: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            encoding=EncodingScheme.ANGLE,
            framework=QuantumFramework.PENNYLANE,
            config=config or {},
        )
        self.encoding_rotation = encoding_rotation.upper()
        self.ansatz = ansatz
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device_name = device_name or "default.qubit"
        self.shots = shots

        # Parameter weights and bias
        self.weights: Optional[pnp.ndarray] = None
        self.bias: Optional[float] = None
        self.training_history: List[Dict[str, float]] = []

        # Setup PennyLane device & QNode
        self.dev = create_quantum_device(wires=self.num_qubits, device_name=self.device_name, shots=self.shots)
        self._init_qnode()

    def _init_qnode(self) -> None:
        """Initialize the PennyLane QNode callable."""
        n_q = self.num_qubits
        depth = self.circuit_depth
        enc_rot = self.encoding_rotation
        ansatz_type = self.ansatz

        # Use backprop on statevector device (e.g. default.qubit) for fast exact analytical gradients
        dev_name_str = getattr(self.dev, "name", "default.qubit")
        diff_method = "backprop" if "default" in dev_name_str else "best"

        @qml.qnode(self.dev, interface="autograd", diff_method=diff_method)
        def _circuit(features, weights):
            return build_vqc_circuit(
                features=features,
                weights=weights,
                num_qubits=n_q,
                circuit_depth=depth,
                encoding_rotation=enc_rot,
                ansatz=ansatz_type,
                measurement_wire=0,
            )

        self._qnode = _circuit

    def _init_parameters(self) -> Tuple[pnp.ndarray, float]:
        """Initialize variational weights with deterministic seed."""
        np.random.seed(self.random_seed)
        param_shape = get_vqc_parameter_shape(self.num_qubits, self.circuit_depth, self.ansatz)
        raw_weights = 0.05 * np.random.randn(*param_shape)
        weights = pnp.array(raw_weights, requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        return weights, bias

    def _cost(self, weights: pnp.ndarray, bias: pnp.ndarray, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Mean squared error loss on PauliZ expectation value with bias using native batch execution."""
        preds = self._qnode(x_batch, weights) + bias
        return pnp.mean((preds - y_batch) ** 2)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "VariationalQuantumClassifier":
        """Train variational parameters on preprocessed features using PennyLane autograd."""
        x_arr = np.asarray(x_train, dtype=np.float64)
        # Convert binary {0, 1} to {-1, +1} for expectation value targets
        y_arr = np.where(np.asarray(y_train) == 1, 1.0, -1.0)

        # Initialize weights and bias
        self.weights, self.bias = self._init_parameters()

        # Select PennyLane optimizer
        if self.optimizer_name == "nesterov":
            opt = qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        elif self.optimizer_name == "gradient_descent":
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        else:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        num_samples = len(x_arr)
        weights = self.weights
        bias = self.bias

        self.training_history = []
        rng = np.random.RandomState(self.random_seed)

        for epoch in range(self.epochs):
            # Mini-batch shuffle
            indices = rng.permutation(num_samples)
            batch_size = min(self.batch_size, num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b = x_arr[batch_idx]
                y_b = y_arr[batch_idx]

                # Optimizer step on weights and bias
                def batch_cost_fn(w, b):
                    return self._cost(w, b, x_b, y_b)

                weights, bias = opt.step(batch_cost_fn, weights, bias)

            # Record epoch training loss
            train_loss = float(self._cost(weights, bias, x_arr, y_arr))
            history_entry = {"epoch": epoch + 1, "train_loss": round(train_loss, 6)}

            if x_val is not None and y_val is not None:
                y_val_pm = np.where(np.asarray(y_val) == 1, 1.0, -1.0)
                val_loss = float(self._cost(weights, bias, np.asarray(x_val, dtype=np.float64), y_val_pm))
                history_entry["val_loss"] = round(val_loss, 6)

            self.training_history.append(history_entry)

        self.weights = weights
        self.bias = float(bias)
        self.is_fitted = True
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Compute raw continuous expectation values with bias: <Z> + bias."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("VQC must be fitted before prediction.")

        x_arr = np.asarray(x, dtype=np.float64)
        if len(x_arr) == 0:
            return np.array([])
        expvals = self._qnode(x_arr, self.weights)
        return np.asarray(expvals) + (self.bias or 0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels {0, 1}."""
        raw_scores = self.predict_raw(x)
        # Class 1 if raw_score >= 0.0 else Class 0
        return np.where(raw_scores >= 0.0, 1, 0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distribution [P(0), P(1)].
        
        Uses affine mapping P(y=1) = np.clip((1 + raw_score) / 2, 0.0, 1.0).
        """
        raw_scores = self.predict_raw(x)
        p1 = np.clip((1.0 + raw_scores) / 2.0, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save_checkpoint(self, filepath: str) -> str:
        """Save quantum variational parameters in .npz format with metadata."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Cannot save unfitted VQC model.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Ensure .npz extension
        if not filepath.endswith(".npz"):
            filepath = f"{os.path.splitext(filepath)[0]}.npz"

        np.savez_compressed(
            filepath,
            weights=np.asarray(self.weights),
            bias=np.array([self.bias if self.bias is not None else 0.0]),
            num_qubits=np.array([self.num_qubits]),
            circuit_depth=np.array([self.circuit_depth]),
            encoding_rotation=np.array([self.encoding_rotation]),
            ansatz=np.array([self.ansatz]),
            random_seed=np.array([self.random_seed]),
        )
        return filepath

    def load_checkpoint(self, filepath: str) -> "VariationalQuantumClassifier":
        """Load quantum variational parameters from .npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"VQC checkpoint not found at: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        self.weights = pnp.array(data["weights"], requires_grad=False)
        self.bias = float(data["bias"][0])
        self.num_qubits = int(data["num_qubits"][0])
        self.circuit_depth = int(data["circuit_depth"][0])
        self.encoding_rotation = str(data["encoding_rotation"][0])
        self.ansatz = str(data["ansatz"][0])
        if "random_seed" in data:
            self.random_seed = int(data["random_seed"][0])

        self.dev = create_quantum_device(wires=self.num_qubits, device_name=self.device_name, shots=self.shots)
        self._init_qnode()
        self.is_fitted = True
        return self

    def export_parameters(self) -> Dict[str, Any]:
        """Export parameters and architectural metadata in structured dictionary."""
        return {
            "model_type": "vqc",
            "framework": "pennylane",
            "num_qubits": self.num_qubits,
            "circuit_depth": self.circuit_depth,
            "ansatz": self.ansatz,
            "encoding_rotation": self.encoding_rotation,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "bias": self.bias,
            "weight_shape": list(self.weights.shape) if self.weights is not None else [],
            "parameter_count": int(np.prod(self.weights.shape) + 1) if self.weights is not None else 0,
            "is_fitted": self.is_fitted,
        }
