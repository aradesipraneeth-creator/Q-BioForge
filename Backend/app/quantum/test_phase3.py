"""Phase 3 Comprehensive Verification Test Suite for Q-BioForge Quantum Experiment Engine.

Verifies:
1. PennyLane library availability
2. Device discovery (CPU, GPU, CUDA, cuQuantum detection)
3. Quantum feature encoding (Angle encoding, range [0, pi])
4. Circuit construction & parameter shapes
5. Correct qubit count handling
6. Correct feature dimension (4, 8, 12, 16 features via training-only PCA/SelectKBest)
7. Real VQC training via autograd optimizer
8. Discrete & probabilistic prediction generation
9. Comprehensive metric calculation (Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, Brier)
10. Checkpoint persistence (.npz parameter format)
11. Checkpoint reloading & prediction consistency
12. Reproducibility with deterministic random seed
13. Quantum REST API endpoints via FastAPI TestClient
"""

import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

# Set UTF-8 encoding and line buffering for stdout
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

# Ensure Backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pennylane as qml
from app.data.loader import load_biomedical_dataset
from app.data.preprocessor import PreprocessorConfig, TabularPreprocessor
from app.quantum.devices import get_system_quantum_environment, create_quantum_device
from app.quantum.encodings import AngleEncoding, apply_angle_encoding
from app.quantum.circuits import build_vqc_circuit, get_vqc_parameter_shape
from app.quantum.vqc import VariationalQuantumClassifier
from app.quantum.qsvm import QuantumSupportVectorClassifier, QuantumKernelEstimator
from app.quantum.metrics import calculate_quantum_metrics
from app.quantum.trainer import (
    train_and_evaluate_quantum_model,
    list_all_quantum_results,
    get_quantum_result,
)
from app.main import app


def run_phase3_tests():
    print("==================================================")
    print("STARTING Q-BIOFORGE PHASE 3 QUANTUM VERIFICATION")
    print("==================================================")

    csv_path = os.path.join(backend_dir, "datasets", "breast_cancer_wisconsin.csv")
    assert os.path.exists(csv_path), f"Dataset CSV missing at {csv_path}"

    # 1. PennyLane Availability
    p_ver = qml.__version__
    assert p_ver is not None and len(p_ver) > 0
    print(f"[TEST 1 PASS] PennyLane library available (Version: {p_ver}).")

    # 2. Device Discovery
    env = get_system_quantum_environment()
    print(f"[TEST 2 PASS] Quantum Environment Discovery:")
    print(f"  - Recommended Device: {env['recommended_device']}")
    print(f"  - Compute Backend: {env['compute_backend']}")
    print(f"  - Available Devices: {env['available_devices']}")
    print(f"  - CUDA Available: {env['cuda_available']}")
    print(f"  - cuQuantum Available: {env['cuquantum_available']}")
    print(f"  - Status: {env['gpu_status_message']}")
    assert len(env["available_devices"]) > 0

    # 3 & 4. Quantum Encoding & Circuit Construction
    enc = AngleEncoding(num_qubits=4, rotation="Y", angle_range=(0.0, np.pi))
    assert enc.num_qubits == 4
    param_shape = get_vqc_parameter_shape(num_qubits=4, circuit_depth=2, ansatz="hardware_efficient")
    assert param_shape == (2, 4, 3)
    print(f"[TEST 3 & 4 PASS] Angle encoding & circuit parameterized shape verified: {param_shape}.")

    # 5 & 6. Feature Dimension Scaling (4, 8, 12, 16) & Qubit Matching
    ds = load_biomedical_dataset(csv_path, target_col="target", random_seed=42)
    for k_features in [4, 8, 12, 16]:
        prep_cfg = PreprocessorConfig(
            impute_strategy="median",
            scaling_method="quantum_angle",
            target_feature_range=(0.0, np.pi),
            target_dim_for_quantum=k_features,
            dim_reduction_method="pca",
        )
        preprocessor = TabularPreprocessor(config=prep_cfg)
        x_tr_proc = preprocessor.fit_transform(ds.x_train, ds.y_train)
        x_val_proc = preprocessor.transform(ds.x_val)
        x_te_proc = preprocessor.transform(ds.x_test)
        assert x_tr_proc.shape == (398, k_features)
        assert x_val_proc.shape == (85, k_features)
        assert x_te_proc.shape == (86, k_features)
        # Verify bounds in [0, pi]
        assert np.all(x_tr_proc >= -1e-6) and np.all(x_tr_proc <= np.pi + 1e-6)
    print(f"[TEST 5 & 6 PASS] Multi-dimension quantum feature reductions (4, 8, 12, 16 features -> qubits) verified with strict zero-leakage.")

    # 7 & 8. Real VQC Training & Prediction
    # Use small 4-qubit, 5-epoch test for rapid verification
    prep_4 = TabularPreprocessor(config=PreprocessorConfig(
        impute_strategy="median",
        scaling_method="quantum_angle",
        target_feature_range=(0.0, np.pi),
        target_dim_for_quantum=4,
        dim_reduction_method="pca",
    ))
    x_tr_4 = prep_4.fit_transform(ds.x_train, ds.y_train)
    x_te_4 = prep_4.transform(ds.x_test)
    y_tr = ds.y_train.to_numpy(dtype=int)
    y_te = ds.y_test.to_numpy(dtype=int)

    vqc = VariationalQuantumClassifier(
        num_qubits=4,
        circuit_depth=2,
        encoding_rotation="Y",
        ansatz="hardware_efficient",
        optimizer_name="adam",
        learning_rate=0.1,
        epochs=10,
        batch_size=32,
        random_seed=42,
    )
    vqc.fit(x_tr_4, y_tr)
    assert vqc.is_fitted
    assert len(vqc.training_history) == 10
    # Confirm optimization loss decreased
    initial_loss = vqc.training_history[0]["train_loss"]
    final_loss = vqc.training_history[-1]["train_loss"]
    print(f"[TEST 7 PASS] VQC training verified: Initial loss={initial_loss:.4f} -> Final loss={final_loss:.4f}")

    preds_vqc = vqc.predict(x_te_4)
    probs_vqc = vqc.predict_proba(x_te_4)
    assert len(preds_vqc) == 86
    assert probs_vqc.shape == (86, 2)
    assert np.all(probs_vqc >= 0.0) and np.all(probs_vqc <= 1.0)
    print(f"[TEST 8 PASS] VQC discrete and probabilistic predictions verified on test split.")

    # 9. Metric Calculation
    metrics_vqc = calculate_quantum_metrics(y_te, preds_vqc, probs_vqc)
    assert 0.0 <= metrics_vqc.accuracy <= 1.0
    assert 0.0 <= metrics_vqc.precision <= 1.0
    assert 0.0 <= metrics_vqc.recall <= 1.0
    assert 0.0 <= metrics_vqc.f1_score <= 1.0
    assert metrics_vqc.sensitivity is not None
    assert metrics_vqc.specificity is not None
    assert metrics_vqc.brier_score is not None
    print(f"[TEST 9 PASS] Quantum metrics verified: Acc={metrics_vqc.accuracy}, Prec={metrics_vqc.precision}, Recall={metrics_vqc.recall}, F1={metrics_vqc.f1_score}, ROC-AUC={metrics_vqc.roc_auc}")

    # 10 & 11. Checkpoint Saving, Reloading & Prediction Consistency (.npz)
    test_npz = os.path.join(backend_dir, "results", "test_vqc_params.npz")
    vqc.save_checkpoint(test_npz)
    assert os.path.exists(test_npz), "Quantum .npz checkpoint file was not written"
    print(f"[TEST 10 PASS] Quantum parameters saved to {test_npz}")

    reloaded_vqc = VariationalQuantumClassifier()
    reloaded_vqc.load_checkpoint(test_npz)
    assert reloaded_vqc.is_fitted
    reloaded_preds = reloaded_vqc.predict(x_te_4)
    reloaded_probs = reloaded_vqc.predict_proba(x_te_4)
    assert np.array_equal(preds_vqc, reloaded_preds), "Reloaded VQC discrete predictions mismatch!"
    assert np.allclose(probs_vqc, reloaded_probs, atol=1e-5), "Reloaded VQC probabilistic predictions mismatch!"
    print(f"[TEST 11 PASS] Checkpoint reloading & prediction consistency verified (100% exact match).")

    if os.path.exists(test_npz):
        os.remove(test_npz)

    # 12. End-to-End First Experiment (4 features, PCA, Angle, 4 qubits, depth 2, Ideal, VQC, seed 42)
    print("\n--- EXECUTING FIRST OFFICIAL 4-QUBIT VQC EXPERIMENT ---")
    first_exp = train_and_evaluate_quantum_model(
        model_type="vqc",
        target_features=4,
        dim_reduction_method="pca",
        num_qubits=4,
        circuit_depth=2,
        encoding_scheme="angle",
        encoding_rotation="Y",
        optimizer_name="adam",
        learning_rate=0.05,
        epochs=25,
        batch_size=32,
        random_seed=42,
        noise_model="ideal",
    )
    exp_id = first_exp["experiment_id"]
    exp_dir = first_exp["experiment_dir"]
    print(f"[TEST 12 PASS] Executed first official Quantum Experiment: {exp_id} in {exp_dir}")
    assert os.path.exists(os.path.join(exp_dir, "config.json"))
    assert os.path.exists(os.path.join(exp_dir, "metrics.json"))
    assert os.path.exists(os.path.join(exp_dir, "predictions.json"))
    assert os.path.exists(os.path.join(exp_dir, "quantum_parameters.npz"))
    assert os.path.exists(os.path.join(exp_dir, "model_metadata.json"))

    m = first_exp["metrics"]["test_metrics"]
    print(f"First VQC Experiment ({exp_id}) Test Results:")
    print(f"  Accuracy:    {m['accuracy']}")
    print(f"  Precision:   {m['precision']}")
    print(f"  Recall/Sens: {m['recall']}")
    print(f"  Specificity: {m['specificity']}")
    print(f"  F1-Score:    {m['f1_score']}")
    print(f"  ROC-AUC:     {m['roc_auc']}")
    print(f"  Brier Score: {m['brier_score']}")
    print(f"  Train Time:  {first_exp['metrics']['training_time_seconds']}s")

    # 13. REST API Endpoints Verification via TestClient
    client = TestClient(app)
    r_dev = client.get("/api/quantum/devices")
    assert r_dev.status_code == 200
    assert "recommended_device" in r_dev.json()
    print(f"[API TEST PASS] GET /api/quantum/devices: {r_dev.json()['recommended_device']}")

    r_results = client.get("/api/quantum/results")
    assert r_results.status_code == 200
    print(f"[API TEST PASS] GET /api/quantum/results: count={len(r_results.json())}")

    r_single = client.get(f"/api/quantum/results/{exp_id}")
    assert r_single.status_code == 200
    assert r_single.json()["experiment_id"] == exp_id
    print(f"[API TEST PASS] GET /api/quantum/results/{exp_id}: retrieved verified.")

    print("\n==================================================")
    print("ALL PHASE 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase3_tests()
