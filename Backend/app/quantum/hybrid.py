"""Hybrid Quantum-Classical Neural Network (Hybrid QML) architecture for Q-BioForge.

Architecture:
Biomedical tabular inputs (e.g. 30 features)
      ↓
Classical Linear Projection (reduces to n_qubits)
      ↓
Quantum Variational Circuit (Angle/Re-uploading Encoding + Entanglement + PauliZ Readout)
      ↓
Classical Dense Output Head (with optional Softmax / Sigmoid calibration)
      ↓
Decision Output {0, 1} and Calibrated Class Probabilities

Supports PyTorch & PennyLane autograd integration.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from app.quantum.base import QuantumClassifier, QuantumFramework, EncodingScheme
from app.quantum.devices import create_quantum_device
from app.quantum.circuits import build_vqc_circuit, get_vqc_parameter_shape


class HybridQuantumClassifier(QuantumClassifier):
    """Hybrid Quantum-Classical Classifier combining classical linear transformations with variational quantum layers."""

    def __init__(
        self,
        num_qubits: int = 4,
        raw_feature_dim: int = 30,
        circuit_depth: int = 2,
        encoding_rotation: str = "Y",
        ansatz: str = "hardware_efficient",
        optimizer_name: str = "adam",
        learning_rate: float = 0.05,
        epochs: int = 25,
        batch_size: int = 32,
        random_seed: int = 42,
        device_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            encoding=EncodingScheme.ANGLE,
            framework=QuantumFramework.PENNYLANE,
            config=config or {},
        )
        self.raw_feature_dim = raw_feature_dim
        self.encoding_rotation = encoding_rotation.upper()
        self.ansatz = ansatz
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device_name = device_name or "default.qubit"

        # Parameters
        self.w_classical_in: Optional[pnp.ndarray] = None   # (raw_feature_dim, num_qubits)
        self.b_classical_in: Optional[pnp.ndarray] = None   # (num_qubits,)
        self.w_quantum: Optional[pnp.ndarray] = None        # (depth, num_qubits, 3)
        self.w_classical_out: Optional[pnp.ndarray] = None  # (1,)
        self.b_classical_out: Optional[pnp.ndarray] = None  # (1,)

        self.training_history: List[Dict[str, float]] = []
        self.dev = create_quantum_device(wires=self.num_qubits, device_name=self.device_name)
        self._init_hybrid_qnode()

    def _init_hybrid_qnode(self) -> None:
        """Initialize the PennyLane QNode for quantum layer execution."""
        n_q = self.num_qubits
        depth = self.circuit_depth
        enc_rot = self.encoding_rotation
        ansatz_type = self.ansatz

        dev_name = getattr(self.dev, "name", "default.qubit")
        diff_method = "backprop" if "default" in dev_name else "best"

        @qml.qnode(self.dev, interface="autograd", diff_method=diff_method)
        def _circuit(latent_features, q_weights):
            return build_vqc_circuit(
                features=latent_features,
                weights=q_weights,
                num_qubits=n_q,
                circuit_depth=depth,
                encoding_rotation=enc_rot,
                ansatz=ansatz_type,
                measurement_wire=0,
            )

        self._qnode = _circuit

    def _init_parameters(self) -> None:
        """Initialize classical projection and quantum variational parameters."""
        np.random.seed(self.random_seed)
        # Classical input projection: (d_in -> n_qubits)
        w_in = 0.1 * np.random.randn(self.raw_feature_dim, self.num_qubits)
        b_in = np.zeros(self.num_qubits)
        # Quantum weights: (depth, n_qubits, 3)
        q_shape = get_vqc_parameter_shape(self.num_qubits, self.circuit_depth, self.ansatz)
        w_q = 0.05 * np.random.randn(*q_shape)
        # Classical output head
        w_out = np.array([1.0])
        b_out = np.array([0.0])

        self.w_classical_in = pnp.array(w_in, requires_grad=True)
        self.b_classical_in = pnp.array(b_in, requires_grad=True)
        self.w_quantum = pnp.array(w_q, requires_grad=True)
        self.w_classical_out = pnp.array(w_out, requires_grad=True)
        self.b_classical_out = pnp.array(b_out, requires_grad=True)

    def _forward(
        self,
        x_batch: np.ndarray,
        w_in: pnp.ndarray,
        b_in: pnp.ndarray,
        w_q: pnp.ndarray,
        w_out: pnp.ndarray,
        b_out: pnp.ndarray,
    ) -> pnp.ndarray:
        """End-to-end forward pass through classical input projection, quantum circuit, and output head."""
        # 1. Classical linear projection with tanh activation mapping to [-pi, pi]
        latent = pnp.dot(x_batch, w_in) + b_in
        angles = pnp.pi * pnp.tanh(latent)

        # 2. Quantum Layer execution
        q_out = self._qnode(angles, w_q)

        # 3. Classical output head
        logits = q_out * w_out[0] + b_out[0]
        return logits

    def _cost(
        self,
        params: List[pnp.ndarray],
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> float:
        """Mean squared error loss on hybrid network predictions."""
        w_in, b_in, w_q, w_out, b_out = params
        logits = self._forward(x_batch, w_in, b_in, w_q, w_out, b_out)
        return pnp.mean((logits - y_batch) ** 2)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HybridQuantumClassifier":
        """Train hybrid classical-quantum parameters using joint gradient descent."""
        x_arr = np.asarray(x_train, dtype=np.float64)
        self.raw_feature_dim = x_arr.shape[1]
        self._init_parameters()

        # Targets in {-1, +1}
        y_arr = np.where(np.asarray(y_train) == 1, 1.0, -1.0)

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        params = [self.w_classical_in, self.b_classical_in, self.w_quantum, self.w_classical_out, self.b_classical_out]

        num_samples = len(x_arr)
        rng = np.random.RandomState(self.random_seed)
        self.training_history = []

        for epoch in range(self.epochs):
            indices = rng.permutation(num_samples)
            batch_size = min(self.batch_size, num_samples)

            for start_idx in range(0, num_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b = x_arr[batch_idx]
                y_b = y_arr[batch_idx]

                def batch_loss_fn(*p):
                    return self._cost(list(p), x_b, y_b)

                params = opt.step(batch_loss_fn, *params)

            # Record epoch losses
            train_loss = float(self._cost(params, x_arr, y_arr))
            history_entry = {"epoch": epoch + 1, "train_loss": round(train_loss, 6)}

            if x_val is not None and y_val is not None:
                y_val_pm = np.where(np.asarray(y_val) == 1, 1.0, -1.0)
                val_loss = float(self._cost(params, np.asarray(x_val, dtype=np.float64), y_val_pm))
                history_entry["val_loss"] = round(val_loss, 6)

            self.training_history.append(history_entry)

        self.w_classical_in, self.b_classical_in, self.w_quantum, self.w_classical_out, self.b_classical_out = params
        self.is_fitted = True
        return self

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Compute continuous hybrid output logits."""
        if not self.is_fitted:
            raise RuntimeError("HybridQuantumClassifier must be fitted before predict.")
        x_arr = np.asarray(x, dtype=np.float64)
        if len(x_arr) == 0:
            return np.array([])
        logits = self._forward(
            x_arr,
            self.w_classical_in,
            self.b_classical_in,
            self.w_quantum,
            self.w_classical_out,
            self.b_classical_out,
        )
        return np.asarray(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels {0, 1}."""
        raw = self.predict_raw(x)
        return np.where(raw >= 0.0, 1, 0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probability distribution [P(0), P(1)]."""
        raw = self.predict_raw(x)
        p1 = np.clip((1.0 + raw) / 2.0, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save_checkpoint(self, filepath: str) -> str:
        """Save hybrid classical-quantum parameters in .npz format with full metadata."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted Hybrid model.")

        if not filepath.endswith(".npz"):
            filepath = f"{os.path.splitext(filepath)[0]}.npz"

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            w_classical_in=np.asarray(self.w_classical_in),
            b_classical_in=np.asarray(self.b_classical_in),
            w_quantum=np.asarray(self.w_quantum),
            w_classical_out=np.asarray(self.w_classical_out),
            b_classical_out=np.asarray(self.b_classical_out),
            num_qubits=np.array([self.num_qubits]),
            raw_feature_dim=np.array([self.raw_feature_dim]),
            circuit_depth=np.array([self.circuit_depth]),
            encoding_rotation=np.array([self.encoding_rotation]),
            ansatz=np.array([self.ansatz]),
            random_seed=np.array([self.random_seed]),
        )
        return filepath

    def load_checkpoint(self, filepath: str) -> "HybridQuantumClassifier":
        """Load hybrid classical-quantum parameters from .npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        self.w_classical_in = pnp.array(data["w_classical_in"], requires_grad=False)
        self.b_classical_in = pnp.array(data["b_classical_in"], requires_grad=False)
        self.w_quantum = pnp.array(data["w_quantum"], requires_grad=False)
        self.w_classical_out = pnp.array(data["w_classical_out"], requires_grad=False)
        self.b_classical_out = pnp.array(data["b_classical_out"], requires_grad=False)

        self.num_qubits = int(data["num_qubits"][0])
        self.raw_feature_dim = int(data["raw_feature_dim"][0])
        self.circuit_depth = int(data["circuit_depth"][0])
        self.encoding_rotation = str(data["encoding_rotation"][0])
        self.ansatz = str(data["ansatz"][0])
        if "random_seed" in data:
            self.random_seed = int(data["random_seed"][0])

        self.dev = create_quantum_device(wires=self.num_qubits, device_name=self.device_name)
        self._init_hybrid_qnode()
        self.is_fitted = True
        return self

    def export_parameters(self) -> Dict[str, Any]:
        """Export parameters and architectural metadata."""
        return {
            "model_type": "hybrid_qml",
            "framework": "pennylane_hybrid",
            "num_qubits": self.num_qubits,
            "raw_feature_dim": self.raw_feature_dim,
            "circuit_depth": self.circuit_depth,
            "ansatz": self.ansatz,
            "encoding_rotation": self.encoding_rotation,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "is_fitted": self.is_fitted,
        }
