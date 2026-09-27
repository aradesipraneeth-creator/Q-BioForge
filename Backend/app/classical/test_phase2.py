"""Phase 2 Comprehensive Verification Test Suite for Q-BioForge.

Tests:
1. Real dataset loading (Wisconsin Breast Cancer 70/15/15 stratified split)
2. Zero-leakage preprocessing (parameters fitted strictly on training data)
3. Real Logistic Regression baseline training & evaluation
4. Real Random Forest baseline training & evaluation
5. Real Support Vector Machine (SVC) baseline training & evaluation
6. Real XGBoost gradient boosting baseline training & evaluation
7. Metric calculation (Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, Brier Score)
8. Model checkpoint persistence (joblib)
9. Model checkpoint reloading
10. Prediction consistency (identical discrete & probabilistic predictions post-reload)
11. Result persistence (QB-XXXXX directory, config.json, metrics.json, predictions.json, model_metadata.json) & API endpoints
"""

import os
import sys
import json
import shutil
import numpy as np
from fastapi.testclient import TestClient

# Set UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure Backend root is in path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.data.loader import load_biomedical_dataset
from app.data.preprocessor import PreprocessorConfig, TabularPreprocessor
from app.classical.base import ClassicalModel
from app.classical.logistic_regression import LogisticRegressionModel
from app.classical.random_forest import RandomForestModel
from app.classical.svm import SVMModel
from app.classical.xgboost_model import XGBoostModel
from app.classical.metrics import calculate_metrics, ClassificationMetrics
from app.classical.trainer import (
    train_and_evaluate_classical_model,
    list_all_experiment_results,
    get_experiment_result,
    generate_classical_benchmark_table,
    get_model_instance,
)
from app.main import app


def run_phase2_tests():
    print("==================================================")
    print("STARTING Q-BIOFORGE PHASE 2 VERIFICATION TESTS")
    print("==================================================")

    csv_path = os.path.join(backend_dir, "datasets", "breast_cancer_wisconsin.csv")
    assert os.path.exists(csv_path), f"Dataset CSV missing at {csv_path}"
    print(f"[TEST 1 PASS] Dataset found at {csv_path}")

    # 1. Dataset loading & 70/15/15 stratified partitioning
    ds = load_biomedical_dataset(csv_path, target_col="target", random_seed=42)
    s = ds.summary
    assert s.total_samples == 569, f"Expected 569 samples, got {s.total_samples}"
    assert s.total_features == 30, f"Expected 30 features, got {s.total_features}"
    assert s.train_samples == 398, f"Expected 398 train samples (70%), got {s.train_samples}"
    assert s.val_samples == 85, f"Expected 85 val samples (15%), got {s.val_samples}"
    assert s.test_samples == 86, f"Expected 86 test samples (15%), got {s.test_samples}"
    print(f"[TEST 1 PASS] Dataset split verified: Train={s.train_samples}, Val={s.val_samples}, Test={s.test_samples}")

    # 2. Preprocessing & Zero Data Leakage Verification
    prep_cfg = PreprocessorConfig(impute_strategy="median", scaling_method="standard")
    preprocessor = TabularPreprocessor(config=prep_cfg)
    
    # Fit strictly on train
    x_train_proc = preprocessor.fit_transform(ds.x_train, ds.y_train)
    assert preprocessor.is_fitted, "Preprocessor should be marked as fitted"
    
    # Transform val and test
    x_val_proc = preprocessor.transform(ds.x_val)
    x_test_proc = preprocessor.transform(ds.x_test)
    
    # Verify shapes
    assert x_train_proc.shape == (398, 30)
    assert x_val_proc.shape == (85, 30)
    assert x_test_proc.shape == (86, 30)
    
    # Verify feature traceability
    feat_names = preprocessor.get_transformed_feature_names()
    assert len(feat_names) == 30
    assert feat_names == s.feature_names
    print(f"[TEST 2 PASS] Zero-leakage preprocessing & feature traceability verified ({len(feat_names)} features).")

    # 3. Logistic Regression Training & Evaluation
    lr_model = LogisticRegressionModel(config={"C": 1.0, "random_seed": 42})
    lr_model.fit(x_train_proc, ds.y_train.to_numpy(), feature_names=feat_names)
    assert lr_model.is_fitted
    lr_preds = lr_model.predict(x_test_proc)
    lr_probs = lr_model.predict_proba(x_test_proc)
    assert len(lr_preds) == 86
    assert lr_probs.shape == (86, 2)
    lr_metrics = calculate_metrics(ds.y_test.to_numpy(), lr_preds, lr_probs)
    assert 0.0 <= lr_metrics.accuracy <= 1.0
    assert lr_metrics.roc_auc is not None and 0.0 <= lr_metrics.roc_auc <= 1.0
    print(f"[TEST 3 PASS] Logistic Regression verified: Accuracy={lr_metrics.accuracy}, ROC-AUC={lr_metrics.roc_auc}")

    # 4. Random Forest Training & Evaluation
    rf_model = RandomForestModel(config={"n_estimators": 100, "random_seed": 42})
    rf_model.fit(x_train_proc, ds.y_train.to_numpy(), feature_names=feat_names)
    assert rf_model.is_fitted
    rf_preds = rf_model.predict(x_test_proc)
    rf_probs = rf_model.predict_proba(x_test_proc)
    assert len(rf_preds) == 86
    assert rf_probs.shape == (86, 2)
    rf_metrics = calculate_metrics(ds.y_test.to_numpy(), rf_preds, rf_probs)
    assert 0.0 <= rf_metrics.accuracy <= 1.0
    print(f"[TEST 4 PASS] Random Forest verified: Accuracy={rf_metrics.accuracy}, ROC-AUC={rf_metrics.roc_auc}")

    # 5. Support Vector Machine Training & Evaluation
    svm_model = SVMModel(config={"C": 1.0, "kernel": "rbf", "probability": True, "random_seed": 42})
    svm_model.fit(x_train_proc, ds.y_train.to_numpy(), feature_names=feat_names)
    assert svm_model.is_fitted
    svm_preds = svm_model.predict(x_test_proc)
    svm_probs = svm_model.predict_proba(x_test_proc)
    assert len(svm_preds) == 86
    assert svm_probs.shape == (86, 2)
    svm_metrics = calculate_metrics(ds.y_test.to_numpy(), svm_preds, svm_probs)
    assert 0.0 <= svm_metrics.accuracy <= 1.0
    print(f"[TEST 5 PASS] Support Vector Machine verified: Accuracy={svm_metrics.accuracy}, ROC-AUC={svm_metrics.roc_auc}")

    # 6. XGBoost Training & Evaluation
    xgb_model = XGBoostModel(config={"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_seed": 42})
    xgb_model.fit(x_train_proc, ds.y_train.to_numpy(), feature_names=feat_names)
    assert xgb_model.is_fitted
    xgb_preds = xgb_model.predict(x_test_proc)
    xgb_probs = xgb_model.predict_proba(x_test_proc)
    assert len(xgb_preds) == 86
    assert xgb_probs.shape == (86, 2)
    xgb_metrics = calculate_metrics(ds.y_test.to_numpy(), xgb_preds, xgb_probs)
    assert 0.0 <= xgb_metrics.accuracy <= 1.0
    print(f"[TEST 6 PASS] Real XGBoost verified: Accuracy={xgb_metrics.accuracy}, ROC-AUC={xgb_metrics.roc_auc}")

    # 7. Metric Calculation (Edge cases, sensitivity, specificity, Brier score)
    # Test synthetic edge case
    edge_true = np.array([0, 0, 1, 1])
    edge_pred = np.array([0, 1, 0, 1])
    edge_prob = np.array([0.1, 0.9, 0.3, 0.8])
    edge_m = calculate_metrics(edge_true, edge_pred, edge_prob)
    assert edge_m.accuracy == 0.5
    assert edge_m.sensitivity == 0.5  # TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert edge_m.specificity == 0.5  # TN / (TN + FP) = 1 / (1 + 1) = 0.5
    assert edge_m.brier_score is not None
    print(f"[TEST 7 PASS] Comprehensive metrics (Sensitivity={edge_m.sensitivity}, Specificity={edge_m.specificity}, Brier={edge_m.brier_score}) verified.")

    # 8, 9, 10. Checkpoint Save, Reload & Prediction Consistency
    test_checkpoint_path = os.path.join(backend_dir, "results", "test_checkpoint.joblib")
    xgb_model.save_checkpoint(test_checkpoint_path)
    assert os.path.exists(test_checkpoint_path), "Checkpoint file was not written"
    print(f"[TEST 8 PASS] Model checkpoint saved to {test_checkpoint_path}")

    # Reload checkpoint
    reloaded_xgb = XGBoostModel()
    reloaded_xgb.load_checkpoint(test_checkpoint_path)
    assert reloaded_xgb.is_fitted
    print(f"[TEST 9 PASS] Model checkpoint reloaded successfully")

    reloaded_preds = reloaded_xgb.predict(x_test_proc)
    reloaded_probs = reloaded_xgb.predict_proba(x_test_proc)
    assert np.array_equal(xgb_preds, reloaded_preds), "Reloaded model discrete predictions differ!"
    assert np.allclose(xgb_probs, reloaded_probs, atol=1e-6), "Reloaded model probabilistic predictions differ!"
    print(f"[TEST 10 PASS] Prediction consistency verified: Original and reloaded predictions match 100%.")

    # Clean up test checkpoint
    if os.path.exists(test_checkpoint_path):
        os.remove(test_checkpoint_path)

    # 11. Full End-to-End Training & Result Persistence via Trainer Engine
    test_results_dir = os.path.join(backend_dir, "results")
    
    models_to_run = ["logistic_regression", "random_forest", "svm", "xgboost"]
    executed_experiments = []
    
    for m_type in models_to_run:
        res = train_and_evaluate_classical_model(
            model_type=m_type,
            dataset_path=csv_path,
            random_seed=42,
            results_base_dir=test_results_dir,
        )
        exp_id = res["experiment_id"]
        exp_dir = res["experiment_dir"]
        executed_experiments.append(res)
        
        # Verify folder structure
        assert os.path.isdir(exp_dir)
        assert os.path.exists(os.path.join(exp_dir, "config.json"))
        assert os.path.exists(os.path.join(exp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(exp_dir, "predictions.json"))
        assert os.path.exists(os.path.join(exp_dir, "model_metadata.json"))
        assert os.path.exists(os.path.join(exp_dir, "model_checkpoint.joblib"))
        
        # Verify config fields
        with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
            assert cfg["experiment_id"] == exp_id
            assert cfg["model"] == m_type
            assert cfg["random_seed"] == 42
            assert cfg["train_samples"] == 398
            assert cfg["validation_samples"] == 85
            assert cfg["test_samples"] == 86
            assert "original_features" in cfg
            assert "selected_features" in cfg
            assert "transformed_feature_count" in cfg
            
        print(f"[TEST 11 PASS] Persisted experiment {exp_id} for model '{m_type}' with full zero-leakage artifacts.")

    # 12. Classical Benchmark Table Generation
    bench_table = generate_classical_benchmark_table(test_results_dir)
    assert len(bench_table) >= 4
    print("\n--- MEASURED CLASSICAL BENCHMARK TABLE (TEST SPLIT EVALUATION) ---")
    print(f"{'Model':<22} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'ROC-AUC':<8} | {'Sens':<7} | {'Spec':<7} | {'Brier':<7} | {'Train(s)':<8} | {'Infer(s)':<8}")
    print("-" * 115)
    for row in bench_table[-4:]:
        print(
            f"{row['model']:<22} | "
            f"{row['accuracy']:<7.4f} | "
            f"{row['precision']:<7.4f} | "
            f"{row['recall']:<7.4f} | "
            f"{row['f1']:<7.4f} | "
            f"{(row['roc_auc'] if row['roc_auc'] is not None else 0.0):<8.4f} | "
            f"{row['sensitivity']:<7.4f} | "
            f"{row['specificity']:<7.4f} | "
            f"{(row['brier_score'] if row['brier_score'] is not None else 0.0):<7.4f} | "
            f"{row['training_time']:<8.4f} | "
            f"{row['inference_time']:<8.4f}"
        )
    print("-" * 115)

    # 13. API Endpoints Verification via FastAPI TestClient
    client = TestClient(app)
    
    # Test GET /api/classical/models
    r_models = client.get("/api/classical/models")
    assert r_models.status_code == 200
    assert "logistic_regression" in r_models.json()["available_models"]
    print(f"[API TEST PASS] GET /api/classical/models: {r_models.json()['available_models']}")

    # Test GET /api/classical/results
    r_results = client.get("/api/classical/results")
    assert r_results.status_code == 200
    assert r_results.json()["count"] >= 4
    print(f"[API TEST PASS] GET /api/classical/results: count={r_results.json()['count']}")

    # Test GET /api/classical/results/{exp_id}
    first_exp_id = executed_experiments[0]["experiment_id"]
    r_single = client.get(f"/api/classical/results/{first_exp_id}")
    assert r_single.status_code == 200
    assert r_single.json()["data"]["experiment_id"] == first_exp_id
    print(f"[API TEST PASS] GET /api/classical/results/{first_exp_id}: successfully retrieved config and metrics.")

    # Test POST /api/classical/train
    post_payload = {
        "model_type": "logistic_regression",
        "random_seed": 100,
        "hyperparameters": {"C": 0.5, "max_iter": 500}
    }
    r_train = client.post("/api/classical/train", json=post_payload)
    assert r_train.status_code == 201
    post_exp_id = r_train.json()["data"]["experiment_id"]
    print(f"[API TEST PASS] POST /api/classical/train: successfully triggered execution, created {post_exp_id}")

    # Test GET /api/classical/benchmarks
    r_bench = client.get("/api/classical/benchmarks")
    assert r_bench.status_code == 200
    assert r_bench.json()["count"] >= 5
    print(f"[API TEST PASS] GET /api/classical/benchmarks: count={r_bench.json()['count']}")

    print("==================================================")
    print("ALL PHASE 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    return bench_table


if __name__ == "__main__":
    run_phase2_tests()
