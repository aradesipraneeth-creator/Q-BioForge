import React, { useState } from 'react';

export function QuantumLabInteractive({ onLaunchExperiment }) {
  const [qubits, setQubits] = useState(8);
  const [encoding, setEncoding] = useState('angle');
  const [depth, setDepth] = useState(3);
  const [noiseModel, setNoiseModel] = useState('depolarizing');
  const [backend, setBackend] = useState('simulator');

  const renderCircuitWires = () => {
    const wireElements = [];
    const maxDisplayQubits = Math.min(qubits, 6);

    for (let q = 0; q < maxDisplayQubits; q++) {
      wireElements.push(
        <div key={q} className="circuit-wire">
          <span className="wire-label">q{q}</span>
          <div className="wire-line">
            ──|0⟩──
            <span className="gate-box theta">
              {encoding === 'angle' ? `Ry(x${q})` : encoding === 'amplitude' ? `Amp(x)` : `ReUp(x${q})`}
            </span>
            ──
            {Array.from({ length: depth }).map((_, dIdx) => (
              <React.Fragment key={dIdx}>
                {q % 2 === 0 ? (
                  <span>──●────────</span>
                ) : (
                  <span>──┼───●────</span>
                )}
                <span className="gate-box">Rz(θ_{q},{dIdx})</span>
                ──
              </React.Fragment>
            ))}
            ───[M] ⟨Z{q}⟩
          </div>
        </div>
      );
    }

    if (qubits > 6) {
      wireElements.push(
        <div key="ellipsis" style={{ color: 'var(--text-dim)', paddingLeft: '36px', fontSize: '11px' }}>
          ⋮ ({qubits - 6} additional qubit lines parameterized across {depth} variational layers)
        </div>
      );
    }

    return wireElements;
  };

  return (
    <section className="quantum-lab-section" id="quantum-lab">
      <div className="editorial-container">
        <div className="section-intro-header">
          <div className="eyebrow">Interactive Circuit Lab</div>
          <h2 className="section-intro-title">Explore representation, circuit depth, and noise.</h2>
          <p style={{ color: 'var(--text-muted)', maxWidth: '640px' }}>
            Simulate parameterized variational quantum classifiers (VQC) and quantum kernels. Tune feature encodings and NISQ noise channels before dispatching campaigns.
          </p>
        </div>

        <div className="lab-layout-grid">
          {/* Controls Panel */}
          <div className="lab-controls-panel">
            <div className="control-group">
              <label className="control-label">Qubit Register Count</label>
              <select 
                className="control-select" 
                value={qubits} 
                onChange={(e) => setQubits(Number(e.target.value))}
              >
                <option value={4}>4 Qubits</option>
                <option value={8}>8 Qubits</option>
                <option value={12}>12 Qubits</option>
                <option value={16}>16 Qubits</option>
                <option value={20}>20 Qubits</option>
                <option value={24}>24 Qubits (HPC Scale)</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Feature Encoding Scheme</label>
              <select 
                className="control-select" 
                value={encoding} 
                onChange={(e) => setEncoding(e.target.value)}
              >
                <option value="angle">Angle Encoding (Ry)</option>
                <option value="amplitude">Amplitude Encoding</option>
                <option value="reuploading">Data Re-uploading</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Variational Depth (Layers)</label>
              <select 
                className="control-select" 
                value={depth} 
                onChange={(e) => setDepth(Number(e.target.value))}
              >
                <option value={1}>Depth 1 (Shallow)</option>
                <option value={2}>Depth 2</option>
                <option value={3}>Depth 3 (Standard VQC)</option>
                <option value={5}>Depth 5</option>
                <option value={8}>Depth 8 (Deep Ansatz)</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">NISQ Noise Model</label>
              <select 
                className="control-select" 
                value={noiseModel} 
                onChange={(e) => setNoiseModel(e.target.value)}
              >
                <option value="ideal">Ideal (Noiseless Statevector)</option>
                <option value="depolarizing">Depolarizing Channel (p = 0.01)</option>
                <option value="readout">Readout Misclassification (p = 0.03)</option>
                <option value="thermal">Thermal Relaxation (T1 = 50μs, T2 = 70μs)</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Simulation Backend</label>
              <select 
                className="control-select" 
                value={backend} 
                onChange={(e) => setBackend(e.target.value)}
              >
                <option value="simulator">Local CPU Simulator (PennyLane/Qiskit)</option>
                <option value="dgx">NVIDIA DGX B200 (cuQuantum GPU Accel.)</option>
              </select>
            </div>

            <button 
              className="btn-coral-primary" 
              style={{ width: '100%', justifyContent: 'center', marginTop: '10px' }}
              onClick={() => onLaunchExperiment({ qubits, encoding, depth, noiseModel, backend })}
            >
              Run Configured Experiment →
            </button>
          </div>

          {/* Circuit Inspection Canvas */}
          <div className="lab-display-panel">
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span>CIRCUIT CANVAS: {qubits}Q / DEPTH {depth}</span>
                <span>PARAMETERS: {qubits * depth} TRAINABLE (θ)</span>
              </div>

              <div className="circuit-inspect-box">
                {renderCircuitWires()}
              </div>
            </div>

            <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: 'var(--border-hairline)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dim)' }}>
                Hardware Target: {backend === 'dgx' ? 'NVIDIA DGX B200 (Classical HPC)' : 'Local Simulator (Classical CPU)'}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--accent-coral)' }}>
                Status: Verified Parameterized Ansatz
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
