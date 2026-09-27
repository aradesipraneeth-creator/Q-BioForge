import React from 'react';

export function ModelRegistrySection({ models = [] }) {
  const count = models.length;

  return (
    <section className="experiments-section" id="models">
      <div className="editorial-container">
        <div style={{ marginBottom: '32px' }}>
          <div className="eyebrow">Checkpoint Artifacts</div>
          <h2 className="section-intro-title">Model registry.</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px', maxWidth: '600px' }}>
            Provenance-tracked checkpoints across classical PyTorch architectures (.pth) and quantum variational parameter arrays (.npz / .json) with full hyperparameter manifests.
          </p>
        </div>

        {count === 0 ? (
          <div className="empty-editorial-box">
            <div className="empty-editorial-title">No trained models in registry</div>
            <p className="empty-editorial-sub">
              Trained checkpoints will be archived in Backend/models/ accompanied by data preprocessing manifests and circuit metadata.
            </p>
          </div>
        ) : (
          <div>
            <div className="editorial-table-head">
              <div>Model ID</div>
              <div>Architecture</div>
              <div>Dataset</div>
              <div>Qubits / Parameters</div>
              <div>Format</div>
              <div>Status</div>
            </div>

            {models.map((m) => (
              <div key={m.model_id} className="editorial-table-row">
                <div style={{ color: 'var(--accent-coral)', fontWeight: 600 }}>{m.model_id}</div>
                <div style={{ color: 'var(--text-main)', fontWeight: 600 }}>{m.model_type}</div>
                <div style={{ color: 'var(--text-muted)' }}>{m.dataset}</div>
                <div>{m.qubit_count ? `${m.qubit_count}Q` : 'Classical Baseline'}</div>
                <div style={{ color: 'var(--text-dim)' }}>{m.checkpoint_format}</div>
                <div>
                  <span className="tag-pill" style={{ borderColor: '#10B981', color: '#10B981' }}>
                    VALIDATED
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
