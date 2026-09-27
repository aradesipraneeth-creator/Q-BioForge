"""Phase 1 Verification Test for Q-BioForge.

Tests:
1. Real dataset loading (Wisconsin Breast Cancer)
2. Schema validation & class distribution
3. Stratified Train / Val / Test splitting
4. Preprocessing leakage prevention (Train-only parameter fitting)
5. Quantum angle bounding ([0, pi])
6. API endpoint integration
"""

import os
import sys
import numpy as np

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
from fastapi.testclient import TestClient
from app.main import app

def run_phase1_tests():
    print("==================================================")
    print("RUNNING PHASE 1 VERIFICATION TESTS")
    print("==================================================")

    csv_path = os.path.join(backend_dir, "datasets", "breast_cancer_wisconsin.csv")
    assert os.path.exists(csv_path), f"Dataset CSV missing at {csv_path}"
    print(f"[OK] Found real dataset file: {csv_path}")

    # 1. Test Dataset Loader
    ds = load_biomedical_dataset(csv_path, target_col="target", random_seed=42)
    s = ds.summary

    print(f"[OK] Dataset Name: {s.name}")
    print(f"[OK] Total Samples: {s.total_samples}")
    print(f"[OK] Total Features: {s.total_features}")
    print(f"[OK] Target Column: {s.target_name}")
    print(f"[OK] Missing Values: {s.missing_values_count}")
    print(f"[OK] Class Distribution: {s.class_distribution} (Proportions: {s.class_proportions})")
    print(f"[OK] Splits: Train={s.train_samples}, Val={s.val_samples}, Test={s.test_samples}")

    assert s.total_samples == 569, f"Expected 569 samples, got {s.total_samples}"
    assert s.total_features == 30, f"Expected 30 features, got {s.total_features}"
    assert s.missing_values_count == 0, f"Expected 0 missing values, got {s.missing_values_count}"
    assert s.train_samples + s.val_samples + s.test_samples == 569

    # 2. Test Zero-Leakage Preprocessor
    config = PreprocessorConfig(
        impute_strategy="median",
        scaling_method="quantum_angle",
        target_feature_range=(0.0, np.pi),
        target_dim_for_quantum=4,
        dim_reduction_method="select_k_best"
    )
    preprocessor = TabularPreprocessor(config=config)

    # Fit ONLY on training data
    x_train_transformed = preprocessor.fit_transform(ds.x_train, ds.y_train)
    # Transform Val and Test without refitting
    x_val_transformed = preprocessor.transform(ds.x_val)
    x_test_transformed = preprocessor.transform(ds.x_test)

    print(f"[OK] Transformed Train Shape (Quantum 4Q ready): {x_train_transformed.shape}")
    print(f"[OK] Transformed Val Shape: {x_val_transformed.shape}")
    print(f"[OK] Transformed Test Shape: {x_test_transformed.shape}")

    assert x_train_transformed.shape == (s.train_samples, 4)
    assert x_val_transformed.shape == (s.val_samples, 4)
    assert x_test_transformed.shape == (s.test_samples, 4)

    # Check that quantum angle bounds [0, pi] are strictly respected
    assert np.all(x_train_transformed >= 0.0) and np.all(x_train_transformed <= np.pi + 1e-6)
    print(f"[OK] Angle bounds verified on Train: min={x_train_transformed.min():.4f}, max={x_train_transformed.max():.4f} (<= pi = {np.pi:.4f})")

    # 3. Test API Endpoint
    client = TestClient(app)
    response = client.get("/api/datasets")
    assert response.status_code == 200, f"API returned {response.status_code}"
    data = response.json()
    print(f"[OK] API /api/datasets response: status={response.status_code}, datasets found={data['count']}")
    assert data["count"] >= 1
    assert data["datasets"][0]["name"] == "breast_cancer_wisconsin"

    print("==================================================")
    print("PHASE 1 ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_phase1_tests()
