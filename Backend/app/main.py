"""Main FastAPI entrypoint for Q-BioForge backend service.

Exposes REST APIs for health monitoring, dataset management, experiment orchestration,
model registry, quantum circuit lab, noise configuration, and reliability analytics.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

# Import API routers
from app.api.routes_health import router as health_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_models import router as models_router
from app.api.routes_benchmarks import router as benchmarks_router
from app.api.routes_reliability import router as reliability_router
from app.api.routes_quantum import router as quantum_router
from app.api.routes_noise import router as noise_router
from app.api.routes_classical import router as classical_router
from app.api.routes_predict import router as predict_router


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Scalable, Noise-Aware Hybrid Quantum-Classical Experimentation Platform for Biomedical Decision Support.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React / Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Top-level health check endpoint for Render health monitoring
@app.get("/health", tags=["Health"])
def root_health():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "compute_backend": settings.COMPUTE_BACKEND,
        "dgx_status": settings.DGX_STATUS,
        "environment": settings.ENVIRONMENT,
    }

# Root endpoint
@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "compute_backend": settings.COMPUTE_BACKEND,
        "dgx_status": settings.DGX_STATUS,
    }

# Register API routers under /api prefix
app.include_router(health_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(benchmarks_router, prefix="/api")
app.include_router(reliability_router, prefix="/api")
app.include_router(quantum_router, prefix="/api")
app.include_router(noise_router, prefix="/api")
app.include_router(classical_router, prefix="/api")
app.include_router(predict_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
