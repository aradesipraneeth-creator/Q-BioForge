import React from 'react';
import { FlaskConical, Clock } from 'lucide-react';
import { Card } from './Card';

export function ExperimentStatusCard({ experiments = [] }) {
  const count = experiments.length;

  return (
    <Card 
      title="Experiment Campaigns" 
      subtitle="Automated parameter grid evaluations" 
      icon={FlaskConical}
    >
      {count === 0 ? (
        <div className="empty-state-box">
          <div className="empty-state-icon">
            <FlaskConical size={28} />
          </div>
          <div className="empty-state-title">0 experiments</div>
          <p className="empty-state-desc">
            Waiting for experiment configuration. Configure dataset, model architecture, qubit count, encoding, and noise profile to launch.
          </p>
        </div>
      ) : (
        <div>
          {/* Future populated experiments list */}
        </div>
      )}
    </Card>
  );
}
