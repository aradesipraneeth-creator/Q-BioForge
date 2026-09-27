"""FastAPI REST API routes for Biomedical Decision Support Prediction and Inference."""

from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.experiments.registry import ModelRegistry
from app.experiments.reliability import ReliabilityEngine, ReliabilityAssessment

router = APIRouter(prefix="/predict", tags=["Inference & Decision Support"])

registry = ModelRegistry()
reliability_engine = ReliabilityEngine()


class PredictionRequest(BaseModel):
    """Input payload for biomedical inference with reliability assessment."""
    model_id: str = Field(..., description="Registered model ID (e.g. 'MOD-QB-00001' or 'MOD-QB-00016')")
    features: List[float] = Field(..., description="Input feature vector (e.g. 4 PCA features or 30 raw features)")


class PredictionResponse(BaseModel):
    """Structured decision support prediction response."""
    prediction: int = Field(..., description="0 for Benign, 1 for Malignant")
    predicted_class: str = Field(..., description="'Benign' or 'Malignant'")
    probabilities: Dict[str, float] = Field(..., description="Calibrated class probabilities")
    confidence: float = Field(..., description="Maximum class probability")
    model_id: str
    model_type: str
    reliability: ReliabilityAssessment
    disclaimer: str = "Biomedical decision support research prototype. Not approved for autonomous clinical diagnosis."


@router.post("", response_model=PredictionResponse)
def execute_prediction(req: PredictionRequest):
    """Execute model inference with automated reliability, shift detection, and decision support state."""
    try:
        # 1. Run prediction with registered model
        pred_res = registry.predict(req.model_id, np.array(req.features))
        probs = np.array([
            pred_res["probabilities"]["class_0_benign"],
            pred_res["probabilities"]["class_1_malignant"]
        ])

        # 2. Assess reliability & multi-model consensus
        reliability = reliability_engine.assess_reliability(
            predicted_probabilities=probs,
            sample_features=np.array(req.features),
        )

        return PredictionResponse(
            prediction=pred_res["prediction"],
            predicted_class=pred_res["predicted_class"],
            probabilities=pred_res["probabilities"],
            confidence=pred_res["confidence"],
            model_id=req.model_id,
            model_type=pred_res["model_type"],
            reliability=reliability,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
