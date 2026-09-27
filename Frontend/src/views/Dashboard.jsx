import React from 'react';
import { 
  FlaskConical, 
  Database, 
  Box, 
  Cpu, 
  Atom, 
  ShieldCheck, 
  Server,
  Layers
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { Card } from '../components/Card';
import { ExperimentStatusCard } from '../components/ExperimentStatusCard';

export function Dashboard({ health, datasets, experiments, models }) {
  const isConnected = health && health.isOnline;
  const isDgx = health?.compute_backend === 'DGX';

  return (
    <div>
      {/* Top Metrics Row */}
      <div className="metrics-grid">
        <MetricCard
          label="Total Experiments"
          value={experiments?.count || "0"}
          subtext="Waiting for experiment"
          icon={FlaskConical}
        />
        <MetricCard
          label="Biomedical Datasets"
          value={datasets?.count || "0"}
          subtext="No datasets loaded"
          icon={Database}
        />
        <MetricCard
          label="Model Registry"
          value={models?.count || "0"}
          subtext="No trained models"
          icon={Box}
        />
        <MetricCard
          label="Compute Backend"
          value={isDgx ? "DGX B200" : "Local"}
          subtext={isDgx ? "HPC cluster connected" : "DGX backend not connected"}
          icon={Cpu}
        />
      </div>

      {/* Main Sections Grid */}
      <div className="dashboard-sections-grid">
        {/* Left Column: Experiments & Quantum Lab Shell */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <ExperimentStatusCard experiments={experiments?.experiments || []} />

          <Card
            title="Quantum Simulation & NISQ Lab"
            subtitle="Configurable circuit simulator & noise channels"
            icon={Atom}
          >
            <table className="infra-table">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Configured Engine</th>
                  <th>Target Hardware</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Framework Support</td>
                  <td className="mono">PennyLane / Qiskit</td>
                  <td>Modular Provider</td>
                </tr>
                <tr>
                  <td>Feature Encodings</td>
                  <td className="mono">Angle, Amplitude, Re-uploading</td>
                  <td>Parametric Circuits</td>
                </tr>
                <tr>
                  <td>Qubit Configurations</td>
                  <td className="mono">4, 8, 12, 16, 20, 24+ (Flexible)</td>
                  <td>Resource-dependent</td>
                </tr>
                <tr>
                  <td>NISQ Noise Channels</td>
                  <td className="mono">Readout, Depolarizing, T1/T2</td>
                  <td>Noise Model Engine</td>
                </tr>
              </tbody>
            </table>

            <div className="info-callout">
              <strong>Noise-Aware Simulation:</strong> Q-BioForge models NISQ gate errors and measurement decoherence to assess algorithmic robustness before physical QPU execution.
            </div>
          </Card>
        </div>

        {/* Right Column: Reliability Framework & Compute Infrastructure */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <Card
            title="Reliability & Decision Support"
            subtitle="Tri-state clinical decision framework"
            icon={ShieldCheck}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#059669' }}>ACCEPT</span>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>High confidence, low uncertainty</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#D97706' }}>REVIEW</span>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Model disagreement / boundary</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-tertiary)', borderRadius: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#DC2626' }}>ABSTAIN</span>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Distribution shift / high noise</span>
              </div>
            </div>

            <div className="info-callout" style={{ borderLeftColor: '#F59E0B' }}>
              <strong>Research Note:</strong> Q-BioForge is designed as a research platform for decision support, not an autonomous diagnostic system.
            </div>
          </Card>

          <Card
            title="Compute Infrastructure"
            subtitle="High-performance & Edge Architecture"
            icon={Server}
          >
            <table className="infra-table">
              <tbody>
                <tr>
                  <td><strong>NVIDIA DGX B200</strong></td>
                  <td>Classical HPC for ML training & quantum simulation</td>
                </tr>
                <tr>
                  <td><strong>Raspberry Pi</strong></td>
                  <td>Edge research & visualization terminal</td>
                </tr>
                <tr>
                  <td><strong>Quantum Backends</strong></td>
                  <td>Hardware-agnostic simulation & future QPU APIs</td>
                </tr>
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </div>
  );
}
