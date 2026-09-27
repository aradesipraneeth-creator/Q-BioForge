import React from 'react';

export function ExperimentsList({ experiments = [], onOpenLauncher }) {
  const count = experiments.length;

  return (
    <section className="experiments-section" id="experiments">
      <div className="editorial-container">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '16px', marginBottom: '32px' }}>
          <div>
            <div className="eyebrow">Experiment Records</div>
            <h2 className="section-intro-title">Recent campaigns.</h2>
          </div>

          <button className="btn-coral-primary" onClick={onOpenLauncher}>
            + New Campaign
          </button>
        </div>

        {count === 0 ? (
          <div className="empty-editorial-box">
            <div className="empty-editorial-title">No experiment campaigns executed yet</div>
            <p className="empty-editorial-sub">
              Awaiting initialization. Configure dataset, variational ansatz, qubit register scale, and noise profile to record deterministic runs.
            </p>
            <button className="btn-coral-primary" onClick={onOpenLauncher}>
              Create First Experiment →
            </button>
          </div>
        ) : (
          <div>
            <div className="editorial-table-head">
              <div>ID</div>
              <div>Model &amp; Ansatz</div>
              <div>Dataset</div>
              <div>Qubits / Noise</div>
              <div>Seed / Prov</div>
              <div>Status</div>
            </div>

            {experiments.map((exp) => (
              <div key={exp.experiment_id} className="editorial-table-row">
                <div style={{ color: 'var(--accent-coral)', fontWeight: 600 }}>{exp.experiment_id}</div>
                <div style={{ color: 'var(--text-main)', fontWeight: 600 }}>{exp.model_type}</div>
                <div style={{ color: 'var(--text-muted)' }}>{exp.dataset}</div>
                <div>{exp.qubit_count ? `${exp.qubit_count}Q · ${exp.noise_model || 'ideal'}` : 'Classical'}</div>
                <div style={{ color: 'var(--text-dim)' }}>Seed: {exp.seed || 42}</div>
                <div>
                  <span className="tag-pill" style={{ borderColor: 'var(--accent-coral)', color: 'var(--accent-coral)' }}>
                    {exp.status || 'QUEUED'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
