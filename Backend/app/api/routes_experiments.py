"""FastAPI REST API routes for Experiment Orchestration and Campaign Management."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.experiments.database import query_experiments, log_audit_event
from app.experiments.campaign import CampaignRunner, MAX_PARALLEL_EXPERIMENTS

router = APIRouter(prefix="/experiments", tags=["Experiments & Campaigns"])


class CampaignLaunchRequest(BaseModel):
    name: str = Field(..., description="Campaign name")
    description: Optional[str] = Field(None, description="Campaign description")
    experiments: List[Dict[str, Any]] = Field(..., description="List of experiment configurations to run")


@router.get("", response_model=List[Dict[str, Any]])
def list_experiments(dataset: Optional[str] = None, model: Optional[str] = None):
    """Query all historical experiments from SQLite database."""
    filters = {}
    if dataset:
        filters["dataset"] = dataset
    if model:
        filters["model_type"] = model
    return query_experiments(filters)


@router.post("/campaigns", response_model=Dict[str, Any])
def launch_campaign(req: CampaignLaunchRequest):
    """Launch a multi-experiment research campaign across classical and quantum configurations."""
    campaign_id = f"QB-CAMP-{int(__import__('time').time()*1000)%100000:05d}"
    runner = CampaignRunner(campaign_id=campaign_id, name=req.name, description=req.description)

    for exp in req.experiments:
        runner.add_experiment(exp)

    log_audit_event(
        action="CAMPAIGN_LAUNCHED",
        experiment_id=campaign_id,
        details=f"Name: {req.name}, Queued: {len(req.experiments)} experiments",
    )

    result = runner.run_campaign(max_parallel=MAX_PARALLEL_EXPERIMENTS)
    return result
