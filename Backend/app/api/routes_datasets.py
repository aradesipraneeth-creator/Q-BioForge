import os
import glob
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.data.loader import load_biomedical_dataset, DatasetSummary

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get("", response_model=Dict[str, Any])
def list_datasets():
    """List available real biomedical tabular datasets with schema and class balance metadata."""
    datasets_dir = settings.DATASETS_DIR
    if not os.path.isabs(datasets_dir):
        # Resolve relative to Backend root
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        datasets_dir = os.path.join(os.path.dirname(backend_root), "datasets")
        if not os.path.exists(datasets_dir):
            datasets_dir = os.path.join(backend_root, "..", "datasets")

    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    if not csv_files:
        # Check backend/datasets
        fallback_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
        csv_files = glob.glob(os.path.join(fallback_dir, "*.csv"))

    summaries: List[Dict[str, Any]] = []

    for csv_file in csv_files:
        try:
            ds = load_biomedical_dataset(csv_file)
            summaries.append(ds.summary.model_dump())
        except Exception as e:
            summaries.append({
                "name": os.path.splitext(os.path.basename(csv_file))[0],
                "filepath": csv_file,
                "error": str(e)
            })

    return {
        "datasets": summaries,
        "count": len(summaries),
        "message": f"Discovered {len(summaries)} real biomedical dataset(s)." if summaries else "No datasets found in datasets directory."
    }
