import React from 'react';

export function ReliabilitySection() {
  return (
    <section className="reliability-section" id="reliability">
      <div className="editorial-container">
        <div className="eyebrow">Clinical Decision Governance</div>
        
        <h2 className="reliability-quote">
          "Not every prediction should be trusted."
        </h2>

        <p style={{ color: 'var(--text-muted)', fontSize: '16px', maxWidth: '680px', lineHeight: 1.6 }}>
          Biomedical machine learning cannot rely purely on aggregate accuracy. Q-BioForge evaluates predictive entropy, Brier calibration, and noise vulnerability to partition inferences into three clinical actions.
        </p>

        <div className="decision-triad-grid">
          {/* ACCEPT */}
          <div className="decision-card accept">
            <div className="decision-state-tag">01 / ACCEPT</div>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '18px', color: 'var(--text-main)' }}>
              High Confidence
            </h3>
            <p className="decision-desc">
              Model probability lies securely within calibrated confidence intervals ($p &gt; 0.90$ or $p &lt; 0.10$) with negligible classical-quantum disagreement and low entropy.
            </p>
            <div className="decision-criteria">
              Criteria: Calibrated Brier Score &lt; 0.08 · In-Distribution Verified
            </div>
          </div>

          {/* REVIEW */}
          <div className="decision-card review">
            <div className="decision-state-tag">02 / REVIEW</div>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '18px', color: 'var(--text-main)' }}>
              Boundary Disagreement
            </h3>
            <p className="decision-desc">
              Sample falls near the non-linear decision boundary or exhibits divergence between classical baselines (e.g., XGBoost) and variational quantum classifiers.
            </p>
            <div className="decision-criteria">
              Criteria: Entropy &gt; 0.65 · Classical/Quantum Margin Divergence
            </div>
          </div>

          {/* ABSTAIN */}
          <div className="decision-card abstain">
            <div className="decision-state-tag">03 / ABSTAIN</div>
            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '18px', color: 'var(--text-main)' }}>
              Safe Abstention
            </h3>
            <p className="decision-desc">
              Severe covariate shift detected, high NISQ noise corruption, or out-of-distribution tabular features render the model's prediction statistically unreliable.
            </p>
            <div className="decision-criteria">
              Criteria: Mahalanobis Distance Shift · Noise Decoherence &gt; Threshold
            </div>
          </div>
        </div>

        <div style={{ marginTop: '36px', padding: '16px 20px', backgroundColor: 'var(--bg-surface)', borderLeft: '2px solid var(--accent-coral)', fontFamily: 'var(--font-mono)', fontSize: '11.5px', color: 'var(--text-muted)' }}>
          <strong>Research Mandate:</strong> Q-BioForge serves as a research platform for clinical decision support and algorithm benchmarking. It is not an autonomous medical diagnostic device.
        </div>
      </div>
    </section>
  );
}
