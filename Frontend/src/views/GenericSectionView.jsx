import React from 'react';
import { Card } from '../components/Card';
import { Database, FlaskConical, Atom, Waves, BarChart3, Box, ShieldCheck, Sparkles } from 'lucide-react';

const SECTION_CONFIG = {
  Datasets: {
    icon: Database,
    title: "Biomedical Datasets",
    subtitle: "Tabular data loader, stratified splitters, and leakage-safe preprocessing",
    emptyTitle: "No datasets loaded",
    emptyDesc: "Upload or place biomedical datasets (CSV/Parquet) in Backend/datasets/. Preprocessing pipelines will isolate training distributions to prevent data leakage."
  },
  Experiments: {
    icon: FlaskConical,
    title: "Experiment Campaigns",
    subtitle: "Automated exploration across Dataset × Model × Qubit Count × Encoding × Noise × Optimizer",
    emptyTitle: "0 experiments",
    emptyDesc: "No active or completed experiment campaigns. Configurable runs will record multi-objective Pareto metrics and reproducibility seeds."
  },
  'Quantum Lab': {
    icon: Atom,
    title: "Quantum Circuit & QML Lab",
    subtitle: "Variational quantum circuits, quantum kernels (QSVM), and hybrid neural networks",
    emptyTitle: "Quantum simulation ready",
    emptyDesc: "Select PennyLane or Qiskit simulator. Configurable qubit counts (4, 8, 12, 16, 20, 24+) execute on local CPU or NVIDIA DGX B200 GPU acceleration."
  },
  'Noise Lab': {
    icon: Waves,
    title: "NISQ Noise Simulation Lab",
    subtitle: "Realistic noise channel profiling (Readout, Depolarizing, Bit-Flip, T1/T2 Relaxation)",
    emptyTitle: "Ideal simulation default",
    emptyDesc: "Select noise profiles to evaluate quantum classifier resilience against decoherence and measurement infidelity."
  },
  Benchmarks: {
    icon: BarChart3,
    title: "Comparative Benchmarks",
    subtitle: "Multi-objective Pareto evaluation across Classical, Quantum, and Hybrid architectures",
    emptyTitle: "No benchmark evaluations recorded",
    emptyDesc: "Benchmarks compare ROC-AUC, F1, Brier calibration, noise robustness, qubit count, circuit depth, and inference cost."
  },
  Models: {
    icon: Box,
    title: "Model Registry",
    subtitle: "Trained classical PyTorch (.pth) and quantum parameter checkpoints",
    emptyTitle: "No trained models in registry",
    emptyDesc: "Trained models will be stored in Backend/models/ with full provenance metadata, hyperparameters, and experiment IDs."
  },
  Reliability: {
    icon: ShieldCheck,
    title: "Reliability & Clinical Decision Support",
    subtitle: "Uncertainty quantification, calibration curves, and tri-state decision framework",
    emptyTitle: "Tri-state decision support engine (ACCEPT / REVIEW / ABSTAIN)",
    emptyDesc: "Evaluates prediction confidence and distribution shift. Q-BioForge is a decision support research tool, not an autonomous diagnostic system."
  },
  Explainability: {
    icon: Sparkles,
    title: "Model Explainability & Sensitivity",
    subtitle: "Classical SHAP feature attribution & quantum circuit parameter sensitivity analysis",
    emptyTitle: "No explanations generated",
    emptyDesc: "Attribution and circuit gradient sensitivity will be computed for selected model checkpoints."
  }
};

export function GenericSectionView({ sectionName }) {
  const config = SECTION_CONFIG[sectionName] || {
    icon: FlaskConical,
    title: sectionName,
    subtitle: "Platform module",
    emptyTitle: "Module initial state",
    emptyDesc: "Section ready for research workflow integration."
  };

  const Icon = config.icon;

  return (
    <div>
      <Card title={config.title} subtitle={config.subtitle} icon={Icon}>
        <div className="empty-state-box">
          <div className="empty-state-icon">
            <Icon size={32} />
          </div>
          <div className="empty-state-title">{config.emptyTitle}</div>
          <p className="empty-state-desc">{config.emptyDesc}</p>
        </div>
      </Card>
    </div>
  );
}
