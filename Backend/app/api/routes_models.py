"""FastAPI REST API routes for Model Registry."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from app.experiments.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Model Registry"])
registry = ModelRegistry()


@router.get("", response_model=List[Dict[str, Any]])
def list_registered_models():
    """Retrieve all verified models registered in Q-BioForge."""
    return registry.list_models()


@router.get("/{model_id}", response_model=Dict[str, Any])
def get_model_details(model_id: str):
    """Retrieve detailed configuration and parameters for a registered model."""
    try:
        model, cfg = registry.load_model(model_id)
        params_meta = getattr(model, "export_parameters", lambda: {})()
        return {
            "model_id": model_id,
            "configuration": cfg,
            "parameters_metadata": params_meta,
            "status": "ready_for_inference",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
