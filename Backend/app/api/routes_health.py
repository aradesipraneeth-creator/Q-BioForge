from fastapi import APIRouter
from app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_health():
    """Health check endpoint to verify backend service connectivity and compute status."""
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "compute_backend": settings.COMPUTE_BACKEND,
        "dgx_status": settings.DGX_STATUS,
        "environment": settings.ENVIRONMENT,
    }
