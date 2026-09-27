"""FastAPI REST API routes for Benchmark Engine and Multi-Objective Pareto Analysis."""

from typing import Any, Dict, List
from fastapi import APIRouter
from app.experiments.benchmarks import generate_unified_benchmark_records, compute_pareto_front

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


@router.get("", response_model=Dict[str, Any])
def get_benchmarks():
    """Retrieve comparative benchmarks across classical and quantum models."""
    records = generate_unified_benchmark_records()
    pareto = compute_pareto_front(records)
    return {
        "benchmarks": records,
        "count": len(records),
        "pareto_analysis": pareto,
        "message": f"Retrieved {len(records)} verified benchmark evaluations." if records else "No benchmark evaluations recorded yet."
    }
