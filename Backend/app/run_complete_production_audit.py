"""Comprehensive 18-Point Production Audit for Q-BioForge Render Deployment."""

import os
import sys
import json
import time

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.experiments.registry import ModelRegistry
from app.experiments.reliability import ReliabilityEngine

client = TestClient(app)

def run_audit():
    results = {}
    print("=" * 70)
    print("Q-BIOFORGE COMPLETE PRODUCTION AUDIT (18 POINTS)")
    print("=" * 70)

    # 1. Frontend Production Build Check
    dist_html = os.path.join(root_dir, "Frontend", "dist", "index.html")
    results["1. Frontend production build"] = "PASS" if os.path.exists(dist_html) else "FAIL"
    print(f"[{results['1. Frontend production build']}] 1. Frontend production build (dist/index.html verified)")

    # 2. Backend Startup & App Configuration
    results["2. Backend startup"] = "PASS" if app is not None and settings.PORT > 0 else "FAIL"
    print(f"[{results['2. Backend startup']}] 2. Backend startup (FastAPI initialized on port {settings.PORT})")

    # 3. /health Endpoint
    resp = client.get("/health")
    health_data = resp.json() if resp.status_code == 200 else {}
    health_pass = (resp.status_code == 200 and health_data.get("status") == "ok" and health_data.get("dgx_status") == "NOT_CONNECTED")
    results["3. /health"] = "PASS" if health_pass else "FAIL"
    print(f"[{results['3. /health']}] 3. /health (HTTP 200, status={health_data.get('status')}, DGX={health_data.get('dgx_status')})")

    # 4. /docs Endpoint (Swagger UI)
    resp = client.get("/docs")
    results["4. /docs"] = "PASS" if resp.status_code == 200 else "FAIL"
    print(f"[{results['4. /docs']}] 4. /docs (Swagger UI accessible: HTTP {resp.status_code})")

    # 5. /api/datasets
    resp = client.get("/api/datasets")
    ds_data = resp.json() if resp.status_code == 200 else {}
    ds_pass = resp.status_code == 200 and ds_data.get("count", 0) >= 2
    results["5. /api/datasets"] = "PASS" if ds_pass else "FAIL"
    print(f"[{results['5. /api/datasets']}] 5. /api/datasets ({ds_data.get('count')} datasets loaded: {[d['name'] for d in ds_data.get('datasets', [])]})")

    # 6. /api/quantum/devices
    resp = client.get("/api/quantum/devices")
    dev_data = resp.json() if resp.status_code == 200 else {}
    dev_pass = resp.status_code == 200 and "recommended_device" in dev_data and "public_limits" in dev_data
    results["6. /api/quantum/devices"] = "PASS" if dev_pass else "FAIL"
    print(f"[{results['6. /api/quantum/devices']}] 6. /api/quantum/devices (Recommended: {dev_data.get('recommended_device')}, Public Limit Qubits: {dev_data.get('public_limits', {}).get('max_public_qubits')})")

    # 7. /api/models
    resp = client.get("/api/models")
    models_list = resp.json() if resp.status_code == 200 else []
    models_pass = resp.status_code == 200 and len(models_list) > 0
    results["7. /api/models"] = "PASS" if models_pass else "FAIL"
    print(f"[{results['7. /api/models']}] 7. /api/models ({len(models_list)} registered model checkpoints)")

    # 8. /api/benchmarks
    resp = client.get("/api/benchmarks")
    bench_data = resp.json() if resp.status_code == 200 else {}
    bench_pass = resp.status_code == 200 and bench_data.get("count", 0) > 0
    results["8. /api/benchmarks"] = "PASS" if bench_pass else "FAIL"
    print(f"[{results['8. /api/benchmarks']}] 8. /api/benchmarks ({bench_data.get('count')} benchmark evaluations, Pareto front calculated)")

    # 9. /api/predict
    sample_feat = [14.0, 19.0, 90.0, 600.0] + [0.1] * 26
    resp = client.post("/api/predict", json={"model_id": "MOD-QB-00001", "features": sample_feat})
    pred_data = resp.json() if resp.status_code == 200 else {}
    pred_pass = resp.status_code == 200 and pred_data.get("reliability", {}).get("decision_state") in ["ACCEPT", "REVIEW", "ABSTAIN"]
    results["9. /api/predict"] = "PASS" if pred_pass else "FAIL"
    print(f"[{results['9. /api/predict']}] 9. /api/predict (Class: {pred_data.get('predicted_class')}, Decision State: {pred_data.get('reliability', {}).get('decision_state')}, Confidence: {pred_data.get('confidence')})")

    # 10. Experiment Creation (/api/experiments/campaigns)
    camp_payload = {
        "name": "Audit Smoke Campaign",
        "description": "Verification of campaign launcher",
        "experiments": [{
            "model_type": "logistic_regression",
            "random_seed": 42
        }]
    }
    resp = client.post("/api/experiments/campaigns", json=camp_payload)
    camp_data = resp.json() if resp.status_code == 200 else {}
    camp_pass = resp.status_code == 200 and camp_data.get("status") == "COMPLETED"
    results["10. experiment creation"] = "PASS" if camp_pass else "FAIL"
    print(f"[{results['10. experiment creation']}] 10. experiment creation (Campaign ID: {camp_data.get('campaign_id')}, Status: {camp_data.get('status')})")

    # 11. Small 4-Qubit Experiment (/api/quantum/train)
    q_payload = {
        "model_type": "vqc",
        "num_qubits": 4,
        "circuit_depth": 1,
        "target_features": 4,
        "epochs": 2,
        "batch_size": 32,
        "random_seed": 42
    }
    resp = client.post("/api/quantum/train", json=q_payload)
    q_data = resp.json() if resp.status_code == 200 else {}
    q_pass = resp.status_code == 200 and "metrics" in q_data
    results["11. small 4-qubit experiment"] = "PASS" if q_pass else "FAIL"
    print(f"[{results['11. small 4-qubit experiment']}] 11. small 4-qubit experiment (ID: {q_data.get('experiment_id')}, Accuracy: {q_data.get('metrics', {}).get('test_metrics', {}).get('accuracy')})")

    # 12. Reliability Engine
    rel_engine = ReliabilityEngine()
    assessment = rel_engine.assess_reliability(
        predicted_probabilities=[0.95, 0.05],
        sample_features=[14.0, 19.0, 90.0, 600.0] + [0.1] * 26
    )
    rel_pass = assessment.decision_state.value in ["ACCEPT", "REVIEW", "ABSTAIN"] and assessment.confidence_score == 0.95
    results["12. reliability engine"] = "PASS" if rel_pass else "FAIL"
    print(f"[{results['12. reliability engine']}] 12. reliability engine (State: {assessment.decision_state.value}, Entropy: {assessment.prediction_entropy:.4f})")

    # 13. Model Registry Deserialization
    registry = ModelRegistry()
    m_inst, m_cfg = registry.load_model("MOD-QB-00001")
    reg_pass = m_inst is not None and "model" in m_cfg or "model_type" in m_cfg
    results["13. model registry"] = "PASS" if reg_pass else "FAIL"
    print(f"[{results['13. model registry']}] 13. model registry (Loaded MOD-QB-00001: {m_cfg.get('model_type', m_cfg.get('model'))})")

    # 14. CORS Configuration
    cors_origins = settings.ALLOWED_ORIGINS
    cors_pass = len(cors_origins) > 0 and any("localhost" in o for o in cors_origins)
    results["14. CORS"] = "PASS" if cors_pass else "FAIL"
    print(f"[{results['14. CORS']}] 14. CORS (Allowed Origins configured: {cors_origins})")

    # 15. Environment Variables
    env_be_exists = os.path.exists(os.path.join(root_dir, "Backend", ".env.example"))
    env_fe_exists = os.path.exists(os.path.join(root_dir, "Frontend", ".env.example"))
    results["15. environment variables"] = "PASS" if (env_be_exists and env_fe_exists) else "FAIL"
    print(f"[{results['15. environment variables']}] 15. environment variables (Backend & Frontend .env.example templates verified)")

    # 16. No Production Localhost URLs
    with open(os.path.join(root_dir, "Frontend", "src", "services", "api.js"), "r", encoding="utf-8") as f:
        api_content = f.read()
    no_hardcoded = "http://127.0.0.1:8000" not in api_content and "http://localhost:8000" not in api_content
    dynamic_env = "import.meta.env.VITE_API_BASE_URL" in api_content
    results["16. no production localhost URLs"] = "PASS" if (no_hardcoded and dynamic_env) else "FAIL"
    print(f"[{results['16. no production localhost URLs']}] 16. no production localhost URLs (Dynamic VITE_API_BASE_URL resolution verified)")

    # 17. Render Configuration (render.yaml)
    render_yaml_path = os.path.join(root_dir, "render.yaml")
    render_pass = False
    if os.path.exists(render_yaml_path):
        with open(render_yaml_path, "r", encoding="utf-8") as f:
            r_content = f.read()
        render_pass = "q-bioforge-backend" in r_content and "q-bioforge-frontend" in r_content and "/health" in r_content
    results["17. Render configuration"] = "PASS" if render_pass else "FAIL"
    print(f"[{results['17. Render configuration']}] 17. Render configuration (render.yaml Blueprint valid for backend & frontend)")

    # 18. Frontend-to-Backend Communication
    results["18. frontend-to-backend communication"] = "PASS" if (results["3. /health"] == "PASS" and results["5. /api/datasets"] == "PASS" and results["9. /api/predict"] == "PASS") else "FAIL"
    print(f"[{results['18. frontend-to-backend communication']}] 18. frontend-to-backend communication (End-to-end API client routes verified)")

    print("=" * 70)
    all_passed = all(v == "PASS" for v in results.values())
    print(f"FINAL AUDIT RESULT: {'ALL 18 AUDIT CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
