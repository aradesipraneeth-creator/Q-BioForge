import React, { useState } from 'react';

export function ExperimentModal({ isOpen, onClose, onLaunch }) {
  const [dataset, setDataset] = useState('breast_cancer_diagnostic');
  const [modelType, setModelType] = useState('VQC');
  const [qubits, setQubits] = useState(8);
  const [encoding, setEncoding] = useState('angle');
  const [noiseModel, setNoiseModel] = useState('depolarizing');
  const [seed, setSeed] = useState(42);
  const [computeTarget, setComputeTarget] = useState('local');

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    onLaunch({
      experiment_id: `QB-${String(Math.floor(Math.random() * 90000) + 10000)}`,
      dataset,
      model_type: modelType,
      qubit_count: modelType.includes('Classical') || modelType === 'XGBoost' ? null : qubits,
      encoding,
      noise_model: noiseModel,
      seed,
      compute_backend: computeTarget,
      status: 'PENDING'
    });
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose}>✕</button>

        <div className="eyebrow">Campaign Launcher</div>
        <h3 className="modal-title">Configure Experiment Run</h3>
        <p className="modal-sub">
          Initialize a reproducible training campaign across classical, quantum, or hybrid architectures.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="modal-form-grid">
            <div className="control-group">
              <label className="control-label">Biomedical Dataset</label>
              <select className="control-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
                <option value="breast_cancer_diagnostic">Wisconsin Breast Cancer (Diagnostic)</option>
                <option value="heart_disease_tabular">Heart Disease Tabular Clinical</option>
                <option value="diabetes_risk_stratification">Diabetes Risk Stratification</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Model Architecture</label>
              <select className="control-select" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                <option value="VQC">Variational Quantum Classifier (VQC)</option>
                <option value="QSVM">Quantum Support Vector Machine (QSVM)</option>
                <option value="Hybrid-QNN">Hybrid Quantum-Classical QNN</option>
                <option value="XGBoost">XGBoost Baseline</option>
                <option value="LogisticRegression">Logistic Regression Baseline</option>
                <option value="PyTorch-MLP">PyTorch MLP Baseline</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Qubits (Quantum models)</label>
              <select className="control-select" value={qubits} onChange={(e) => setQubits(Number(e.target.value))}>
                <option value={4}>4 Qubits</option>
                <option value={8}>8 Qubits</option>
                <option value={12}>12 Qubits</option>
                <option value={16}>16 Qubits</option>
                <option value={24}>24 Qubits (HPC Scale)</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Feature Encoding</label>
              <select className="control-select" value={encoding} onChange={(e) => setEncoding(e.target.value)}>
                <option value="angle">Angle Encoding (Ry)</option>
                <option value="amplitude">Amplitude Encoding</option>
                <option value="reuploading">Data Re-uploading</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">NISQ Noise Model</label>
              <select className="control-select" value={noiseModel} onChange={(e) => setNoiseModel(e.target.value)}>
                <option value="ideal">Ideal Simulation</option>
                <option value="depolarizing">Depolarizing Channel</option>
                <option value="readout">Readout Misclassification</option>
                <option value="thermal">Thermal T1/T2 Relaxation</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Random Seed</label>
              <input 
                type="number" 
                className="control-input" 
                value={seed} 
                onChange={(e) => setSeed(Number(e.target.value))} 
              />
            </div>
          </div>

          <div className="control-group" style={{ marginBottom: '24px' }}>
            <label className="control-label">Compute Infrastructure Target</label>
            <select className="control-select" value={computeTarget} onChange={(e) => setComputeTarget(e.target.value)}>
              <option value="local">Local Development Runtime (CPU)</option>
              <option value="DGX">NVIDIA DGX B200 (Classical GPU Cluster)</option>
            </select>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button type="button" className="btn-outline" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn-coral-primary">
              Initialize Campaign →
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
