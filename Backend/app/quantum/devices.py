"""Device and simulator discovery engine for PennyLane & Qiskit in Q-BioForge.

Provides programmatic detection of:
- PennyLane version
- Available CPU simulators (e.g., 'default.qubit', 'lightning.qubit')
- GPU / CUDA availability (e.g., 'lightning.gpu')
- cuQuantum SDK presence
- Compute hardware context (Local CPU vs NVIDIA DGX B200 HPC)
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
import pennylane as qml

_CACHED_ENV: Optional[Dict[str, Any]] = None


def get_system_quantum_environment() -> Dict[str, Any]:
    """Inspect and report the local quantum simulation environment honestly."""
    global _CACHED_ENV
    if _CACHED_ENV is not None:
        return _CACHED_ENV

    pennylane_version = qml.__version__

    # 1. Inspect installed PennyLane devices
    available_devices: List[str] = []
    for dev_name in ["default.qubit", "lightning.qubit", "lightning.gpu", "qiskit.aer", "default.mixed"]:
        try:
            # Check if device plugin is recognized
            dev = qml.device(dev_name, wires=1)
            available_devices.append(dev_name)
        except Exception:
            pass

    # 2. Check CUDA / GPU availability
    cuda_available = False
    gpu_name = None
    gpu_count = 0
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        # PyTorch not installed; check for CUDA environment indicators
        pass

    # 3. Check cuQuantum availability
    cuquantum_available = False
    try:
        import cuquantum  # type: ignore
        cuquantum_available = True
    except ImportError:
        cuquantum_available = False

    # 4. Check Qiskit availability
    qiskit_available = False
    qiskit_version = None
    try:
        import qiskit
        qiskit_available = True
        qiskit_version = getattr(qiskit, "__version__", "installed")
    except ImportError:
        qiskit_available = False

    # 5. Determine active execution backend
    if "lightning.gpu" in available_devices and cuda_available:
        compute_backend = "NVIDIA DGX / GPU Accelerated (lightning.gpu)"
        recommended_device = "lightning.gpu"
        gpu_status_message = f"GPU acceleration active: {gpu_name or 'CUDA Device'} ({gpu_count} GPU(s))"
    elif "lightning.qubit" in available_devices:
        compute_backend = "High-Performance CPU C++ (lightning.qubit)"
        recommended_device = "lightning.qubit"
        gpu_status_message = "GPU quantum simulation unavailable in current environment. Using lightning.qubit CPU simulator."
    else:
        compute_backend = "Standard State-Vector CPU (default.qubit)"
        recommended_device = "default.qubit"
        gpu_status_message = "GPU quantum simulation unavailable in current environment. Using default.qubit CPU simulator."

    env_dict = {
        "pennylane_version": pennylane_version,
        "available_devices": available_devices,
        "recommended_device": recommended_device,
        "compute_backend": compute_backend,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "cuquantum_available": cuquantum_available,
        "qiskit_available": qiskit_available,
        "qiskit_version": qiskit_version,
        "gpu_status_message": gpu_status_message,
    }
    _CACHED_ENV = env_dict
    return env_dict


def create_quantum_device(
    wires: int,
    device_name: Optional[str] = None,
    shots: Optional[int] = None,
) -> qml.devices.Device:
    """Instantiate a validated PennyLane quantum device with fallback to default.qubit."""
    env = get_system_quantum_environment()

    if device_name is None:
        target_name = env["recommended_device"]
    else:
        target_name = device_name

    try:
        if shots is not None and shots > 0:
            return qml.device(target_name, wires=wires, shots=shots)
        return qml.device(target_name, wires=wires)
    except Exception as e:
        # Fallback to standard CPU default.qubit if requested device fails
        fallback_name = "default.qubit"
        if shots is not None and shots > 0:
            return qml.device(fallback_name, wires=wires, shots=shots)
        return qml.device(fallback_name, wires=wires)
