"""Compute backend abstraction for local simulation vs. DGX B200 HPC orchestration."""

import abc
from typing import Any, Dict, Optional
from app.config import settings


class ComputeBackend(abc.ABC):
    """Abstract base class for Q-BioForge compute execution backends."""

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    @abc.abstractmethod
    def get_status_info(self) -> Dict[str, Any]:
        pass


class LocalComputeBackend(ComputeBackend):
    """Local or Render in-process CPU state-vector quantum simulation backend."""

    @property
    def backend_name(self) -> str:
        return "render-local" if settings.COMPUTE_BACKEND == "render-local" else "local"

    @property
    def is_available(self) -> bool:
        return True

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "type": "In-Process CPU Simulation",
            "status": "ACTIVE",
            "max_qubits": settings.MAX_PUBLIC_QUBITS,
            "max_depth": settings.MAX_PUBLIC_DEPTH,
            "description": "Lightweight web API & CPU state-vector simulator for demo workloads and validation.",
        }


class DGXComputeBackend(ComputeBackend):
    """Classical High-Performance Computing (HPC) cluster backend (NVIDIA DGX B200)."""

    @property
    def backend_name(self) -> str:
        return "DGX"

    @property
    def is_available(self) -> bool:
        return bool(settings.COMPUTE_BACKEND == "DGX" and settings.DGX_ENDPOINT)

    def get_status_info(self) -> Dict[str, Any]:
        if self.is_available:
            return {
                "backend": "DGX",
                "type": "Classical High-Performance Computing Cluster",
                "status": "CONNECTED",
                "endpoint": settings.DGX_ENDPOINT,
                "role": "Classical ML training, PyTorch deep learning, GPU quantum simulation, and experiment campaigns.",
            }
        return {
            "backend": "DGX",
            "type": "Classical High-Performance Computing Cluster",
            "status": "NOT_CONNECTED",
            "note": "DGX B200 is available as college HPC infrastructure but is not connected to the public Render deployment.",
            "role": "Classical ML training, PyTorch deep learning, GPU quantum simulation, and experiment campaigns.",
        }


def get_active_compute_backend() -> ComputeBackend:
    """Return active compute backend instance based on environment settings."""
    if settings.COMPUTE_BACKEND == "DGX":
        return DGXComputeBackend()
    return LocalComputeBackend()
