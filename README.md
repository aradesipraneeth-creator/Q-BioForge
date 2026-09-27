# Q-BioForge

> **A Research-Grade Hybrid Quantum-Classical Machine Learning Platform for Biomedical Decision Support.**

---

## 1. Research Disclaimer & Deployment Architecture

> [!IMPORTANT]
> **Scientific Integrity & Infrastructure Roles**:
> - **Public Render Deployment**: Render hosts the public web application (`q-bioforge-frontend` static site) and REST API service (`q-bioforge-backend`). Render executes lightweight API dispatch, preprocessed inference, and safe small-scale state-vector simulation ($N_{\text{qubits}} \le 8$). **Render is not a quantum computer and is not the DGX cluster.**
> - **NVIDIA DGX B200 HPC Cluster**: The college's NVIDIA DGX B200 cluster is a **classical high-performance computing system** dedicated to training deep neural networks, executing GPU-accelerated quantum state-vector simulations (via `lightning.gpu` / `cuQuantum`), and orchestrating large parameter exploration campaigns. **The DGX B200 is not a quantum computer.**
> - **Raspberry Pi Edge Terminal**: The Raspberry Pi is utilized as an **edge research and visualization terminal** for local clinical monitors and telemetry display. **It is not a quantum computer.**
> - **Clinical Disclaimer**: Q-BioForge is an experimental research prototype for clinical decision support and algorithm benchmarking. It does **not** constitute an autonomous medical diagnostic system.

---

## 2. Target Production Deployment (Render)

The platform is configured for zero-downtime deployment on Render via the infrastructure blueprint [render.yaml](file:///c:/Users/arade/OneDrive/Documents/Q%20Bio%20Forge/render.yaml):

```
                       PUBLIC INTERNET (HTTPS)
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
   RENDER STATIC SITE                           RENDER WEB SERVICE
  `q-bioforge-frontend`                         `q-bioforge-backend`
   (React 18 + Vite SPA)                         (FastAPI + Python 3.11)
        │                                               │
        │─── HTTPS REST API (VITE_API_BASE_URL) ───────▶│
                                                        │
                                            ┌───────────┴───────────┐
                                            │                       │
                                      CPU INFERENCE           DGX GATEWAY
                                     & PUBLIC GUARDS          (COLLEGE HPC)
                                      (Qubits <= 8)          (When Connected)
```

### Services Specification

| Service Name | Type | Runtime | Build Command | Start Command / Publish Dir |
| :--- | :--- | :--- | :--- | :--- |
| **`q-bioforge-backend`** | Web Service | Python 3.11 | `pip install -r requirements.txt` | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **`q-bioforge-frontend`** | Static Site | Node.js (Vite) | `npm install && npm run build` | `dist` (with rewrite rule `/*` $\to$ `/index.html`) |

---

## 3. Environment Configuration

### Backend Environment Variables (`Backend/.env.example`)

| Variable | Description | Production Default |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Deployment environment mode | `production` |
| `HOST` | Server bind host address | `0.0.0.0` |
| `PORT` | Server bind port (Render dynamic) | `10000` / `$PORT` |
| `FRONTEND_URL` | Deployed frontend origin for CORS | `https://q-bioforge-frontend.onrender.com` |
| `COMPUTE_BACKEND` | Active compute backend mode | `render-local` |
| `MAX_PUBLIC_QUBITS` | Safety guard for public demo execution | `8` |
| `MAX_PUBLIC_DEPTH` | Maximum circuit depth for public demo | `4` |
| `MAX_PUBLIC_EXPERIMENTS`| Maximum parallel public experiments | `2` |
| `MAX_PUBLIC_RUNTIME_SECONDS` | Timeout for public simulations | `60` |
| `DATABASE_PATH` | Path to SQLite database file | `results/q_bioforge.db` |

### Frontend Environment Variables (`Frontend/.env.example`)

| Variable | Description | Production Value |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Production FastAPI backend URL | `https://q-bioforge-backend.onrender.com` |

---

## 4. Deploying to Render

### Option A: One-Click Render Blueprint (Recommended)
1. Fork or push the repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New** $\to$ **Blueprint**.
3. Connect your repository containing [render.yaml](file:///c:/Users/arade/OneDrive/Documents/Q%20Bio%20Forge/render.yaml).
4. Render will automatically provision both the `q-bioforge-backend` Web Service and `q-bioforge-frontend` Static Site with all environment variables and routing rules linked.

### Option B: Manual Service Creation
1. **Backend Web Service**:
   - **Runtime**: Python
   - **Root Directory**: `Backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
2. **Frontend Static Site**:
   - **Runtime**: Static
   - **Root Directory**: `Frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
   - **Rewrite Rule**: `/*` $\to$ `/index.html`
   - **Environment Variable**: `VITE_API_BASE_URL = https://<your-backend-service>.onrender.com`

---

## 5. Local Development & Verification

### Running Locally
```bash
# 1. Start Backend Server
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Start Frontend Server
cd ../Frontend
npm install
npm run dev
```

### Running Test Suites
```bash
# Verify all research phases (classical ML, quantum circuits, noise channels, reliability)
python Backend/app/test_all_phases.py

# Verify Render production deployment readiness & public guards
python Backend/app/test_deployment_readiness.py
```

---

## 6. Public API Reference

- `GET /health` — Production health check (returns HTTP 200, compute backend, and honest DGX connection status).
- `GET /docs` — Interactive OpenAPI / Swagger UI.
- `GET /api/datasets` — Discovered tabular biomedical datasets (Wisconsin Breast Cancer & UCI Cleveland Heart Disease).
- `GET /api/quantum/devices` — Discovered PennyLane simulation devices, GPU availability, and public safety limits.
- `GET /api/models` — Registered model checkpoints in the Model Registry.
- `GET /api/benchmarks` — Classical vs. quantum benchmark evaluations and multi-objective Pareto front.
- `POST /api/predict` — Biomedical decision support inference with Shannon predictive entropy, 2-sample KS distribution shift analysis, multi-model consensus, and `ACCEPT` / `REVIEW` / `ABSTAIN` state classification.
- `POST /api/quantum/train` — Variational Quantum Classifier (VQC) / QSVM training with automatic `RESOURCE_LIMITED` public guards.
