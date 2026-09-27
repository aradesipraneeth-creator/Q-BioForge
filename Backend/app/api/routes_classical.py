"""FastAPI router for Classical Machine Learning baselines in Q-BioForge."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.classical.trainer import (
    train_and_evaluate_classical_model,
    list_all_experiment_results,
    get_experiment_result,
    generate_classical_benchmark_table,
    MODEL_REGISTRY,
)


router = APIRouter(prefix="/classical", tags=["Classical Baselines"])


class TrainClassicalRequest(BaseModel):
    """Request payload for training a real classical baseline model."""
    model_type: str = Field(..., description="Model identifier: 'logistic_regression', 'random_forest', 'svm', or 'xgboost'")
    dataset_path: Optional[str] = Field(None, description="Optional path to dataset CSV file. Defaults to Wisconsin Breast Cancer.")
    target_col: str = Field("target", description="Target column label in dataset.")
    random_seed: int = Field(42, description="Deterministic random seed for partitioning and training.")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom model hyperparameters.")
    preprocessing_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom preprocessing configurations.")


@router.get("/models", response_model=Dict[str, Any])
def list_available_classical_models():
    """List supported classical baseline model architectures."""
    return {
        "available_models": list(MODEL_REGISTRY.keys()),
        "description": "Deterministic, zero-leakage classical ML baselines for biomedical decision support."
    }


@router.post("/train", status_code=status.HTTP_201_CREATED, response_model=Dict[str, Any])
def train_classical_model(request: TrainClassicalRequest):
    """Train, validate, and test a real classical baseline model with full artifact persistence."""
    try:
        result = train_and_evaluate_classical_model(
            model_type=request.model_type,
            dataset_path=request.dataset_path,
            target_col=request.target_col,
            random_seed=request.random_seed,
            hyperparameters=request.hyperparameters,
            preprocessing_config=request.preprocessing_config,
        )
        return {
            "status": "success",
            "message": f"Successfully trained {request.model_type} baseline model.",
            "data": result,
        }
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fe))
    except ImportError as ie:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ie))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Training failed: {str(e)}")


@router.get("/results", response_model=Dict[str, Any])
def get_all_results():
    """Retrieve all stored classical baseline experiment runs."""
    results = list_all_experiment_results()
    return {
        "results": results,
        "count": len(results),
    }


@router.get("/results/{experiment_id}", response_model=Dict[str, Any])
def get_result_by_id(experiment_id: str):
    """Retrieve full artifacts (config, metrics, predictions, metadata) for a single experiment."""
    result = get_experiment_result(experiment_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found in results repository."
        )
    return {
        "status": "success",
        "data": result,
    }


@router.get("/benchmarks", response_model=Dict[str, Any])
def get_classical_benchmarks():
    """Retrieve complete classical benchmark table across all executed models."""
    table = generate_classical_benchmark_table()
    return {
        "benchmarks": table,
        "count": len(table),
    }
