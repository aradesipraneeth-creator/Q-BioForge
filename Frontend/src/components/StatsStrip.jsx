import React from 'react';

export function StatsStrip({ experiments, models, datasets }) {
  const formatNumber = (num, digits = 3) => {
    if (num === undefined || num === null || num === 0) {
      return '0'.repeat(digits);
    }
    return String(num).padStart(digits, '0');
  };

  return (
    <section className="stats-strip">
      <div className="editorial-container">
        <div className="stats-grid">
          <div className="stat-cell">
            <span className="stat-label">Experiments</span>
            <div className="stat-number">
              {formatNumber(experiments?.count, 3)}
            </div>
            <span className="stat-sub">
              {experiments?.count > 0 ? 'Recorded runs' : 'Awaiting runs'}
            </span>
          </div>

          <div className="stat-cell">
            <span className="stat-label">Models</span>
            <div className="stat-number">
              {formatNumber(models?.count, 3)}
            </div>
            <span className="stat-sub">
              {models?.count > 0 ? 'Trained checkpoints' : 'Registry empty'}
            </span>
          </div>

          <div className="stat-cell">
            <span className="stat-label">Datasets</span>
            <div className="stat-number">
              {formatNumber(datasets?.count, 2)}
            </div>
            <span className="stat-sub">Tabular benchmarks</span>
          </div>

          <div className="stat-cell">
            <span className="stat-label">Qubit Scale</span>
            <div className="stat-number highlight">
              4 → N
            </div>
            <span className="stat-sub">Hardware-agnostic</span>
          </div>

          <div className="stat-cell">
            <span className="stat-label">Noise Models</span>
            <div className="stat-number">
              06
            </div>
            <span className="stat-sub">NISQ error channels</span>
          </div>
        </div>
      </div>
    </section>
  );
}
