/**
 * API Service for communicating with the Q-BioForge FastAPI backend.
 * Supports configurable VITE_API_BASE_URL for Render production deployment.
 */

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || '';
export const API_BASE = rawBaseUrl.replace(/\/+$/, '');

/**
 * Check backend health status.
 * @returns {Promise<{ status: string, service?: string, compute_backend?: string, version?: string, isOnline: boolean }>}
 */
export async function fetchHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4500);

    const response = await fetch(`${API_BASE}/health`, {
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      return {
        status: 'error',
        isOnline: false,
        error: `HTTP ${response.status}`,
      };
    }

    const data = await response.json();
    return {
      ...data,
      isOnline: data.status === 'ok',
    };
  } catch (err) {
    return {
      status: 'offline',
      service: 'q-bioforge-backend',
      isOnline: false,
      error: err.message || 'Connection failed',
    };
  }
}

/**
 * Fetch datasets inventory.
 */
export async function fetchDatasets() {
  try {
    const response = await fetch(`${API_BASE}/api/datasets`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    return { datasets: [], count: 0, message: 'No datasets loaded' };
  }
}

/**
 * Fetch experiments list from SQLite.
 */
export async function fetchExperiments() {
  try {
    const response = await fetch(`${API_BASE}/api/experiments`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return {
      experiments: Array.isArray(data) ? data : (data.experiments || []),
      count: Array.isArray(data) ? data.length : (data.count || 0),
    };
  } catch (err) {
    return { experiments: [], count: 0, message: '0 experiments' };
  }
}

/**
 * Fetch model registry checkpoints.
 */
export async function fetchModels() {
  try {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return {
      models: Array.isArray(data) ? data : (data.models || []),
      count: Array.isArray(data) ? data.length : (data.count || 0),
    };
  } catch (err) {
    return { models: [], count: 0, message: 'No trained models' };
  }
}

/**
 * Fetch unified classical and quantum benchmark comparisons and Pareto frontier.
 */
export async function fetchBenchmarks() {
  try {
    const response = await fetch(`${API_BASE}/api/benchmarks`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    return { total_models: 0, benchmarks: [], pareto_frontier: [] };
  }
}

/**
 * Fetch quantum simulation devices and compute info.
 */
export async function fetchQuantumDevices() {
  try {
    const response = await fetch(`${API_BASE}/api/quantum/devices`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    return { devices: [], default_device: 'default.qubit' };
  }
}

/**
 * Trigger quantum model training.
 */
export async function launchQuantumTrain(payload) {
  const response = await fetch(`${API_BASE}/api/quantum/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Quantum training failed' }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return await response.json();
}

/**
 * Trigger campaign execution.
 */
export async function launchCampaign(payload) {
  const response = await fetch(`${API_BASE}/api/experiments/campaigns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Campaign launch failed' }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return await response.json();
}

/**
 * Run decision-support prediction and reliability governance.
 */
export async function predictSample(payload) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return await response.json();
}

/**
 * Evaluate NISQ noise channels.
 */
export async function evaluateNoise(payload) {
  const response = await fetch(`${API_BASE}/api/noise/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Noise evaluation failed' }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return await response.json();
}
