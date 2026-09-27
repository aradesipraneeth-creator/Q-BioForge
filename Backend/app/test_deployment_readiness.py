"""Comprehensive Deployment Readiness Verification Test for Q-BioForge on Render."""

import os
import sys
import json

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_deployment():
    print("=================================================================")
    print("Q-BIOFORGE DEPLOYMENT READINESS VERIFICATION")
    print("=================================================================")

    # 1. Health Check Endpoint (Root /health)
    resp = client.get("/health")
    assert resp.status_code == 200, f"/health failed: {resp.text}"
    health_data = resp.json()
    assert health_data["status"] == "ok"
    assert health_data["service"] == "q-bioforge-backend"
    assert health_data["dgx_status"] == "NOT_CONNECTED"
    print(f"[PASS] /health -> Status: {health_data['status']}, Backend: {health_data['compute_backend']}, DGX: {health_data['dgx_status']}")

    # 2. Root Index Endpoint
    resp = client.get("/")
    assert resp.status_code == 200
    print(f"[PASS] GET / -> Service: {resp.json()['service']}, Docs: {resp.json()['docs']}")

    # 3. Datasets API
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    ds_data = resp.json()
    assert ds_data["count"] >= 1
    print(f"[PASS] GET /api/datasets -> Discovered {ds_data['count']} datasets: {[d['name'] for d in ds_data['datasets']]}")

    # 4. Quantum Devices & Public Resource Limits API
    resp = client.get("/api/quantum/devices")
    assert resp.status_code == 200
    dev_data = resp.json()
    assert "public_limits" in dev_data
    print(f"[PASS] GET /api/quantum/devices -> Recommended device: {dev_data['recommended_device']}, Public limit qubits: {dev_data['public_limits']['max_public_qubits']}")

    # 5. Model Registry API
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models_data = resp.json()
    assert isinstance(models_data, list)
    print(f"[PASS] GET /api/models -> {len(models_data)} registered checkpoints found.")

    # 6. Benchmarks API
    resp = client.get("/api/benchmarks")
    assert resp.status_code == 200
    bench_data = resp.json()
    pareto_pts = bench_data.get("pareto_analysis", {}).get("pareto_frontier", [])
    print(f"[PASS] GET /api/benchmarks -> Total models: {bench_data['count']}, Pareto points: {len(pareto_pts)}")

    # 7. Real Prediction & Reliability Endpoint
    sample_features = [14.0, 19.0, 90.0, 600.0] + [0.1] * 26
    pred_payload = {
        "model_id": "MOD-QB-00001",
        "features": sample_features
    }
    resp = client.post("/api/predict", json=pred_payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    pred_res = resp.json()
    rel_state = pred_res["reliability"]["decision_state"]
    assert rel_state in ["ACCEPT", "REVIEW", "ABSTAIN"]
    print(f"[PASS] POST /api/predict -> Prediction: {pred_res['predicted_class']}, State: {rel_state}, Confidence: {pred_res['confidence']:.4f}")

    # 8. Public Resource Safety Guard Test (Exceeding Qubits)
    heavy_train_payload = {
        "model_type": "vqc",
        "num_qubits": 20, # Exceeds MAX_PUBLIC_QUBITS (8)
        "circuit_depth": 2,
        "target_features": 4,
    }
    resp = client.post("/api/quantum/train", json=heavy_train_payload)
    assert resp.status_code == 200
    guard_res = resp.json()
    assert guard_res["status"] == "RESOURCE_LIMITED"
    print(f"[PASS] Resource Guard: 20-qubit request gracefully returned -> {guard_res['status']}")

    # 9. Small Real Quantum Experiment Test (Safe 4-qubit VQC on CPU)
    safe_train_payload = {
        "model_type": "vqc",
        "num_qubits": 4,
        "circuit_depth": 1,
        "target_features": 4,
        "epochs": 2,
        "batch_size": 32,
        "random_seed": 42
    }
    resp = client.post("/api/quantum/train", json=safe_train_payload)
    assert resp.status_code == 200
    train_res = resp.json()
    assert "metrics" in train_res
    print(f"[PASS] Safe 4-Qubit Experiment -> Experiment ID: {train_res.get('experiment_id')}, Status: {train_res.get('status')}")

    print("=================================================================")
    print("ALL RENDER DEPLOYMENT READINESS TESTS PASSED!")
    print("=================================================================")

if __name__ == "__main__":
    test_deployment()
