import React from 'react';
import { 
  LayoutDashboard, 
  Database, 
  FlaskConical, 
  Atom, 
  Waves, 
  BarChart3, 
  Box, 
  ShieldCheck, 
  Sparkles,
  Server
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'Dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'Datasets', label: 'Datasets', icon: Database },
  { id: 'Experiments', label: 'Experiments', icon: FlaskConical },
  { id: 'Quantum Lab', label: 'Quantum Lab', icon: Atom },
  { id: 'Noise Lab', label: 'Noise Lab', icon: Waves },
  { id: 'Benchmarks', label: 'Benchmarks', icon: BarChart3 },
  { id: 'Models', label: 'Models', icon: Box },
  { id: 'Reliability', label: 'Reliability', icon: ShieldCheck },
  { id: 'Explainability', label: 'Explainability', icon: Sparkles },
];

export function Sidebar({ activeSection, onSelectSection, health }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-badge">
          <div className="brand-icon">Q</div>
          <div>
            <div className="brand-title">Q-BioForge</div>
            <div className="brand-subtitle">Research Platform</div>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Navigation</div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectSection(item.id)}
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div>HPC Target:</div>
        <div className="infra-status">
          <Server size={13} />
          <span>{health?.compute_backend === 'DGX' ? 'DGX B200 (Active)' : 'DGX backend not connected'}</span>
        </div>
      </div>
    </aside>
  );
}
