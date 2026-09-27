import React from 'react';

export function Footer({ onOpenLauncher }) {
  return (
    <>
      <section className="closing-section">
        <div className="editorial-container">
          <div className="closing-card">
            <div className="eyebrow" style={{ color: '#FF705E' }}>
              Experimentation Engine
            </div>
            <h2 className="closing-title">Run the experiment.</h2>
            <p className="closing-desc">
              Explore how feature representation, qubit scale, and NISQ noise affect biomedical model calibration and decision reliability.
            </p>
            <button className="btn-coral-primary" onClick={onOpenLauncher} style={{ marginTop: '12px' }}>
              Launch Experiment Campaign →
            </button>
          </div>
        </div>
      </section>

      <footer className="editorial-footer">
        <div className="editorial-container">
          <div className="footer-inner">
            <div>
              <strong>Q-BIOFORGE</strong> · Hybrid Quantum-Classical Platform for Biomedical Decision Support
            </div>
            <div>
              NVIDIA DGX B200 (Classical HPC) · Raspberry Pi (Edge Terminal)
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
