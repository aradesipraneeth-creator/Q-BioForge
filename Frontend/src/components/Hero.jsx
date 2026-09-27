import React, { useState } from 'react';

export function Hero({ onOpenLauncher, onNavigateToLab }) {
  const [activeGate, setActiveGate] = useState(null);

  return (
    <section className="hero-section" id="top">
      <div className="editorial-container">
        <div className="hero-grid">
          {/* Left Column: Editorial Display Typography */}
          <div>
            <div className="eyebrow">
              Hybrid Quantum × Biomedical Research
            </div>

            <h1 className="hero-headline">
              Engineering reliable <em>quantum learning</em> for biomedical data.
            </h1>

            <p className="hero-description">
              A scalable, noise-aware experimentation platform investigating quantum representation, parameter scaling, and NISQ noise robustness across high-stakes decision boundaries.
            </p>

            <div className="hero-actions">
              <button className="btn-coral-primary" onClick={onOpenLauncher}>
                Launch Experiment →
              </button>
              <button className="btn-outline" onClick={onNavigateToLab}>
                Explore Quantum Lab
              </button>
            </div>

            {/* Pipeline Flow */}
            <div className="pipeline-flow">
              <span className="flow-node active-node">Classical Data</span>
              <span className="flow-arrow">→</span>
              <span className="flow-node active-node">Quantum Encoding</span>
              <span className="flow-arrow">→</span>
              <span className="flow-node">Hybrid Model</span>
              <span className="flow-arrow">→</span>
              <span className="flow-node">NISQ Noise</span>
              <span className="flow-arrow">→</span>
              <span className="flow-node active-node">Reliability Triad</span>
            </div>
          </div>

          {/* Right Column: Abstract Quantum Circuit Schematic */}
          <div className="quantum-schematic-card">
            <div className="schematic-header">
              <span>Ansatz: Variational Feature Map</span>
              <span>U(θ, x) · |0⟩⊗n</span>
            </div>

            <div className="circuit-canvas">
              {/* Qubit Wire 0 */}
              <div className="circuit-wire">
                <span className="wire-label">q0</span>
                <div className="wire-line">
                  ──|0⟩──
                  <span 
                    className="gate-box theta"
                    onMouseEnter={() => setActiveGate('Ry(x0)')}
                    onMouseLeave={() => setActiveGate(null)}
                  >
                    Ry(x₀)
                  </span>
                  ──●──────────────●───────
                  <span className="gate-box">Rz(θ₀)</span>
                  ───[M] ⟨Z₀⟩
                </div>
              </div>

              {/* Qubit Wire 1 */}
              <div className="circuit-wire">
                <span className="wire-label">q1</span>
                <div className="wire-line">
                  ──|0⟩──
                  <span 
                    className="gate-box theta"
                    onMouseEnter={() => setActiveGate('Ry(x1)')}
                    onMouseLeave={() => setActiveGate(null)}
                  >
                    Ry(x₁)
                  </span>
                  ──┼────●─────────┼───●───
                  <span className="gate-box">Rz(θ₁)</span>
                  ───[M] ⟨Z₁⟩
                </div>
              </div>

              {/* Qubit Wire 2 */}
              <div className="circuit-wire">
                <span className="wire-label">q2</span>
                <div className="wire-line">
                  ──|0⟩──
                  <span 
                    className="gate-box theta"
                    onMouseEnter={() => setActiveGate('Ry(x2)')}
                    onMouseLeave={() => setActiveGate(null)}
                  >
                    Ry(x₂)
                  </span>
                  ──●────┼────●────●───┼───
                  <span className="gate-box">Rz(θ₂)</span>
                  ───[M] ⟨Z₂⟩
                </div>
              </div>

              {/* Qubit Wire 3 */}
              <div className="circuit-wire">
                <span className="wire-label">q3</span>
                <div className="wire-line">
                  ──|0⟩──
                  <span 
                    className="gate-box theta"
                    onMouseEnter={() => setActiveGate('Ry(x3)')}
                    onMouseLeave={() => setActiveGate(null)}
                  >
                    Ry(x₃)
                  </span>
                  ───────●────┼────────●───
                  <span className="gate-box">Rz(θ₃)</span>
                  ───[M] ⟨Z₃⟩
                </div>
              </div>
            </div>

            <div className="schematic-footer">
              <span>{activeGate ? `Inspecting: ${activeGate}` : 'Entanglement: Ring CNOT'}</span>
              <span>4 Qubits / Depth 3</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
