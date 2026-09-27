"""FastAPI REST API routes for Reliability and Clinical Decision Support Evaluation."""

from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.experiments.reliability import ReliabilityEngine, ReliabilityAssessment

router = APIRouter(prefix="/reliability", tags=["Reliability & Decision Support"])
engine = ReliabilityEngine()


class ReliabilityCheckRequest(BaseModel):
    probabilities: List[float] = Field(..., description="Predicted class probabilities [P(0), P(1)]")
    features: List[float] = Field(..., description="Input feature vector")


@router.get("", response_model=Dict[str, Any])
def get_reliability_status():
    """Retrieve reliability and decision support status."""
    return {
        "framework": "Biomedical Decision Support (Tri-state: ACCEPT / REVIEW / ABSTAIN)",
        "disclaimer": "Q-BioForge is a research platform for decision support, not an autonomous medical diagnostic device.",
        "thresholds": {
            "confidence_accept": engine.conf_accept,
            "confidence_abstain": engine.conf_abstain,
            "agreement_accept": engine.agree_accept,
            "agreement_abstain": engine.agree_abstain,
            "ks_test_p_value": engine.ks_thresh,
        },
        "states_supported": ["ACCEPT", "REVIEW", "ABSTAIN"],
    }


@router.post("/assess", response_model=ReliabilityAssessment)
def assess_sample_reliability(req: ReliabilityCheckRequest):
    """Assess reliability, confidence, calibration, and shift for a specific sample."""
    try:
        res = engine.assess_reliability(
            predicted_probabilities=np.array(req.probabilities),
            sample_features=np.array(req.features),
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
