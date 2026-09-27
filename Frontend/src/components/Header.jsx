import React from 'react';

export function Header({ health, activeSection, onSelectSection, onOpenLauncher }) {
  const isOnline = health && health.isOnline;

  const NAV_ITEMS = [
    { id: 'research', label: 'Research' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'models', label: 'Models' },
    { id: 'quantum-lab', label: 'Quantum Lab' },
    { id: 'reliability', label: 'Reliability' },
  ];

  return (
    <header className="editorial-header">
      <div className="header-inner">
        <a 
          href="#top" 
          className="brand-link"
          onClick={(e) => {
            e.preventDefault();
            onSelectSection('home');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <span className="brand-dot"></span>
          <span>Q-BioForge</span>
        </a>

        <nav className="nav-links">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-link-btn ${activeSection === item.id ? 'active' : ''}`}
              onClick={() => onSelectSection(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="header-right">
          <div className="health-indicator" title={`Backend: ${isOnline ? 'Online' : 'Offline'}`}>
            <span className={`health-dot ${isOnline ? 'live' : 'off'}`}></span>
            <span>{isOnline ? 'Connected' : 'Offline'}</span>
          </div>

          <button className="btn-coral-primary" onClick={onOpenLauncher}>
            Run Experiment
          </button>
        </div>
      </div>
    </header>
  );
}
