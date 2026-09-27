import React from 'react';

export function MetricCard({ label, value, subtext, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        {Icon && (
          <div className="metric-icon-box">
            <Icon size={16} />
          </div>
        )}
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtext">{subtext}</div>
    </div>
  );
}
