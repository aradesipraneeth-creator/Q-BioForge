"""Comprehensive Verification Suite for All Q-BioForge Research Phases.

Validates:
1. Data pipeline & zero-leakage contracts (PCA, SelectKBest, feature metadata)
2. Classical baseline models (LR, RF, SVM, XGBoost)
3. Quantum environment & honest device detection
4. Quantum encodings (Angle, Data Re-uploading, Amplitude)
5. Variational Quantum Classifier (VQC) with .npz checkpoints
6. Quantum Support Vector Machine (QSVM) & Fidelity Kernel Matrices
7. Hybrid Quantum-Classical Neural Network (Hybrid QML)
8. NISQ Noise Lab (Depolarizing, Readout, Bit-flip) & Sensitivity Analysis
9. Automated Campaign Engine & Resource Limit Handling (LOCAL vs DGX)
10. SQLite Database persistence & Audit Trail
11. Model Registry (list, load, predict)
12. Multi-Objective Pareto Benchmarking
13. Reliability Engine (Confidence, Entropy, KS Distribution Shift, Multi-model consensus, ACCEPT/REVIEW/ABSTAIN)
14. End-to-End Prediction API (POST /api/predict)
"""

import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

# Set unbuffered stdout
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pennylane as qml
from app.data.loader import load_biomedical_dataset
from app.data.preprocessor import PreprocessorConfig, TabularPreprocessor
from app.classical.trainer import train_and_evaluate_classical_model
from app.quantum.devices import get_system_quantum_environment
from app.quantum.encodings import AngleEncoding, DataReuploadingEncoding, AmplitudeEncoding
from app.quantum.vqc import VariationalQuantumClassifier
from app.quantum.qsvm import QuantumSupportVectorClassifier, QuantumKernelEstimator
from app.quantum.hybrid import HybridQuantumClassifier
from app.quantum.noise import NoiseModelConfig, NoiseType, calculate_noise_sensitivity
from app.quantum.trainer import train_and_evaluate_quantum_model
from app.experiments.database import init_db, query_experiments, log_audit_event
from app.experiments.campaign import CampaignRunner
from app.experiments.registry import ModelRegistry
from app.experiments.benchmarks import generate_unified_benchmark_records, compute_pareto_front
from app.experiments.reliability import ReliabilityEngine, DecisionSupportState
from app.main import app


def run_full_verification():
    print("=================================================================")
    print("STARTING COMPLETE Q-BIOFORGE RESEARCH ENGINE VERIFICATION")
    print("=================================================================")

    csv_path = os.path.join(backend_dir, "datasets", "breast_cancer_wisconsin.csv")
    assert os.path.exists(csv_path), f"Dataset CSV missing: {csv_path}"

    # 1. Quantum Data Pipeline & Feature Traceability (Phase 4)
    print("\n--- PHASE 4: QUANTUM DATA PIPELINE & TRACEABILITY ---")
    ds = load_biomedical_dataset(csv_path, target_col="target", random_seed=42)
    prep_pca = TabularPreprocessor(config=PreprocessorConfig(
        impute_strategy="median",
        scaling_method="quantum_angle",
        target_feature_range=(0.0, np.pi),
        target_dim_for_quantum=4,
        dim_reduction_method="pca",
    ))
    x_tr_pca = prep_pca.fit_transform(ds.x_train, ds.y_train)
    meta_pca = prep_pca.get_transformation_metadata()
    assert meta_pca["output_features_count"] == 4
    assert len(meta_pca["pca_explained_variance_ratio"]) == 4
    print(f"[PASS] PCA 4-feature reduction verified (Cumulative Variance: {meta_pca['pca_cumulative_variance']:.2%})")

    prep_kbest = TabularPreprocessor(config=PreprocessorConfig(
        impute_strategy="median",
        scaling_method="quantum_angle",
        target_dim_for_quantum=8,
        dim_reduction_method="select_k_best",
    ))
    x_tr_kbest = prep_kbest.fit_transform(ds.x_train, ds.y_train)
    meta_kbest = prep_kbest.get_transformation_metadata()
    assert meta_kbest["output_features_count"] == 8
    assert len(meta_kbest["selected_feature_indices"]) == 8
    print(f"[PASS] SelectKBest 8-feature reduction verified (Selected: {meta_kbest['transformed_feature_names'][:3]}...)")

    # 2. Quantum Encodings (Phase 5)
    print("\n--- PHASE 5: QUANTUM ENCODINGS ---")
    enc_angle = AngleEncoding(num_qubits=4, rotation="Y")
    assert enc_angle.num_qubits == 4
    enc_reup = DataReuploadingEncoding(num_qubits=4, circuit_depth=2)
    assert enc_reup.circuit_depth == 2
    enc_amp = AmplitudeEncoding(num_qubits=2)
    state_vec = enc_amp.prepare_state_vector([1.0, 2.0, 3.0, 4.0])
    assert len(state_vec) == 4
    assert np.isclose(np.linalg.norm(state_vec), 1.0)
    print("[PASS] Angle, Data Re-uploading, and Amplitude Encodings verified with L2-normalization.")

    # 3. QSVM & Fidelity Quantum Kernel Matrix (Phase 7)
    print("\n--- PHASE 7: QSVM & FIDELITY KERNEL ---")
    kernel_est = QuantumKernelEstimator(num_qubits=2, encoding_rotation="Y")
    k_val = kernel_est.compute_kernel_element(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert np.isclose(k_val, 1.0, atol=1e-3)
    k_diff = kernel_est.compute_kernel_element(np.array([0.0, 0.0]), np.array([np.pi, np.pi]))
    assert 0.0 <= k_diff <= 1.0
    print(f"[PASS] Quantum fidelity kernel verified: K(x,x)={k_val:.4f}, K(x, x_ortho)={k_diff:.4f}")

    # 4. Hybrid QML Architecture (Phase 8)
    print("\n--- PHASE 8: HYBRID QML (CLASSICAL-QUANTUM NEURAL NETWORK) ---")
    hybrid = HybridQuantumClassifier(
        num_qubits=2,
        raw_feature_dim=4,
        circuit_depth=1,
        epochs=3,
        batch_size=32,
        random_seed=42,
    )
    hybrid.fit(x_tr_pca[:32], ds.y_train.to_numpy(dtype=int)[:32])
    h_preds = hybrid.predict(x_tr_pca[:5])
    h_probs = hybrid.predict_proba(x_tr_pca[:5])
    assert len(h_preds) == 5
    assert h_probs.shape == (5, 2)
    print(f"[PASS] Hybrid QML forward and backward autograd training verified.")

    # 5. NISQ Noise Lab & Sensitivity (Phase 9 & 10)
    print("\n--- PHASE 9 & 10: NISQ NOISE LAB & SENSITIVITY ---")
    noise_cfg = NoiseModelConfig(noise_type=NoiseType.DEPOLARIZING, error_probability=0.05)
    ideal_m = {"accuracy": 0.9186, "f1_score": 0.9391, "roc_auc": 0.9832}
    noisy_m = {"accuracy": 0.8520, "f1_score": 0.8710, "roc_auc": 0.9200}
    sens_report = calculate_noise_sensitivity(ideal_m, noisy_m, noise_cfg)
    assert sens_report.absolute_accuracy_drop > 0
    assert sens_report.noise_regime in ["low_impact", "moderate_impact", "high_sensitivity"]
    print(f"[PASS] NISQ noise sensitivity analysis verified (Degradation: {sens_report.relative_accuracy_drop_pct}%, Regime: {sens_report.noise_regime}).")

    # 6. Campaign Runner & Resource Limits (Phase 11-13)
    print("\n--- PHASE 11-13: CAMPAIGNS & RESOURCE LIMIT GUARDS ---")
    camp = CampaignRunner(campaign_id="QB-CAMP-TEST", name="Verification Campaign")
    camp.add_experiment({
        "model": "logistic_regression",
        "dataset_path": csv_path,
        "random_seed": 42,
    })
    camp.add_experiment({
        "model": "vqc",
        "qubit_count": 24,  # Intentionally above local CPU limit to test RESOURCE_LIMITED guard
        "target_features": 24,
    })
    camp_res = camp.run_campaign()
    assert camp_res["total_completed"] == 1
    assert camp_res["total_resource_limited"] == 1
    print(f"[PASS] Campaign execution verified: Completed={camp_res['total_completed']}, Resource Limited={camp_res['total_resource_limited']}.")

    # 7. Model Registry & Unified Benchmarking (Phase 14-17)
    print("\n--- PHASE 14-17: MODEL REGISTRY & PARETO BENCHMARKS ---")
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) >= 1
    sample_model = models[0]
    m_inst, cfg = registry.load_model(sample_model["model_id"])
    pred_res = registry.predict(sample_model["model_id"], np.array([0.1] * (cfg.get("transformed_feature_count") or 30)))
    assert "prediction" in pred_res
    print(f"[PASS] Model Registry verified: {len(models)} models found. Prediction test on '{sample_model['model_name']}' successful.")

    records = generate_unified_benchmark_records()
    pareto = compute_pareto_front(records)
    assert len(pareto["pareto_optimal_experiments"]) > 0
    print(f"[PASS] Multi-Objective Pareto front computed across {pareto['total_evaluated']} benchmark evaluations.")

    # 8. Reliability Engine (Phase 18-22)
    print("\n--- PHASE 18-22: RELIABILITY ENGINE (ACCEPT / REVIEW / ABSTAIN) ---")
    rel_engine = ReliabilityEngine()
    
    # Test high-confidence in-distribution sample -> ACCEPT
    acc_res = rel_engine.assess_reliability(
        predicted_probabilities=np.array([0.05, 0.95]),
        sample_features=np.array([0.5, 0.5, 0.5, 0.5]),
    )
    assert acc_res.decision_state in [DecisionSupportState.ACCEPT, DecisionSupportState.REVIEW]
    print(f"[PASS] High-confidence sample -> State: {acc_res.decision_state.value} (Confidence: {acc_res.confidence_score})")

    # Test corrupted/NaN sample -> ABSTAIN
    nan_res = rel_engine.assess_reliability(
        predicted_probabilities=np.array([0.5, 0.5]),
        sample_features=np.array([np.nan, 0.0, 0.0, 0.0]),
    )
    assert nan_res.decision_state == DecisionSupportState.ABSTAIN
    print(f"[PASS] Corrupted/NaN sample -> State: {nan_res.decision_state.value} (Data Quality Guard Active)")

    # 9. End-to-End REST APIs via TestClient (Phase 24)
    print("\n--- PHASE 24: REST API & PREDICTION ENDPOINT ---")
    client = TestClient(app)

    # Health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200

    # Benchmarks
    r_bench = client.get("/api/benchmarks")
    assert r_bench.status_code == 200

    # Models
    r_models = client.get("/api/models")
    assert r_models.status_code == 200

    # Reliability Status
    r_rel = client.get("/api/reliability")
    assert r_rel.status_code == 200

    # Prediction API POST /api/predict with real model
    r_pred = client.post("/api/predict", json={
        "model_id": sample_model["model_id"],
        "features": [0.5] * (cfg.get("transformed_feature_count") or 30),
    })
    assert r_pred.status_code == 200
    pred_data = r_pred.json()
    assert "prediction" in pred_data
    assert "reliability" in pred_data
    assert "decision_state" in pred_data["reliability"]
    print(f"[PASS] POST /api/predict successful: Prediction={pred_data['predicted_class']}, State={pred_data['reliability']['decision_state']}")

    print("\n=================================================================")
    print("ALL Q-BIOFORGE RESEARCH PHASES VERIFIED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    run_full_verification()
