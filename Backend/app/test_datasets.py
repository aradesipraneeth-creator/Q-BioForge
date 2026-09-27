"""Test dataset discovery and multi-dataset loading."""
import os
import sys

# Ensure Backend is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.data.loader import load_biomedical_dataset

def main():
    datasets_dir = os.path.join(backend_dir, "datasets")
    for fname in os.listdir(datasets_dir):
        if fname.endswith(".csv"):
            fpath = os.path.join(datasets_dir, fname)
            ds = load_biomedical_dataset(fpath)
            print(f"[DATASET] {fname} -> Samples: {ds.summary.total_samples}, Features: {ds.summary.total_features}, Dist: {ds.summary.class_distribution}")

if __name__ == "__main__":
    main()
