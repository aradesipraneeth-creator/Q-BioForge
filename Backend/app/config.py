import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings loaded from environment variables for Q-BioForge."""
    PROJECT_NAME: str = "q-bioforge-backend"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    
    # Host & Port (Render dynamically assigns PORT)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Compute backend: 'render-local', 'local' (CPU development) or 'DGX' (high-performance compute infrastructure)
    COMPUTE_BACKEND: str = os.getenv("COMPUTE_BACKEND", "render-local" if os.getenv("RENDER") else "local")
    DGX_ENDPOINT: Optional[str] = os.getenv("DGX_ENDPOINT", None)
    
    # Compute status reporting
    @property
    def DGX_STATUS(self) -> str:
        if self.COMPUTE_BACKEND == "DGX" and self.DGX_ENDPOINT:
            return "CONNECTED"
        return "NOT_CONNECTED"
    
    # Public Render runtime safeguards & resource limits
    MAX_PUBLIC_QUBITS: int = int(os.getenv("MAX_PUBLIC_QUBITS", "8"))
    MAX_PUBLIC_DEPTH: int = int(os.getenv("MAX_PUBLIC_DEPTH", "4"))
    MAX_PUBLIC_EXPERIMENTS: int = int(os.getenv("MAX_PUBLIC_EXPERIMENTS", "2"))
    MAX_PUBLIC_RUNTIME_SECONDS: int = int(os.getenv("MAX_PUBLIC_RUNTIME_SECONDS", "60"))
    
    # CORS Configuration
    FRONTEND_URL: Optional[str] = os.getenv("FRONTEND_URL", None)
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
        configured = os.getenv("ALLOWED_ORIGINS", "")
        if configured:
            origins.extend([o.strip() for o in configured.split(",") if o.strip()])
        if self.FRONTEND_URL:
            clean_url = self.FRONTEND_URL.strip().rstrip("/")
            if clean_url not in origins:
                origins.append(clean_url)
        return list(dict.fromkeys(origins))
    
    # Paths & storage
    DATABASE_PATH: Optional[str] = os.getenv("DATABASE_PATH", None)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)
    
    DATASETS_DIR: str = os.getenv("DATASETS_DIR", "datasets")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")

settings = Settings()
