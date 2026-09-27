import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { StatsStrip } from './components/StatsStrip';
import { ResearchSections } from './components/ResearchSections';
import { QuantumLabInteractive } from './components/QuantumLabInteractive';
import { ReliabilitySection } from './components/ReliabilitySection';
import { ComputeInfrastructure } from './components/ComputeInfrastructure';
import { ExperimentsList } from './components/ExperimentsList';
import { ModelRegistrySection } from './components/ModelRegistrySection';
import { Footer } from './components/Footer';
import { ExperimentModal } from './components/ExperimentModal';
import { fetchHealth, fetchDatasets, fetchExperiments, fetchModels } from './services/api';

export function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [health, setHealth] = useState({ isOnline: false, status: 'checking', compute_backend: 'local' });
  const [datasets, setDatasets] = useState({ datasets: [], count: 0 });
  const [experiments, setExperiments] = useState({ experiments: [], count: 0 });
  const [models, setModels] = useState({ models: [], count: 0 });
  const [isModalOpen, setIsModalOpen] = useState(false);

  const loadPlatformData = useCallback(async () => {
    try {
      const [healthRes, datasetsRes, expRes, modelsRes] = await Promise.all([
        fetchHealth(),
        fetchDatasets(),
        fetchExperiments(),
        fetchModels(),
      ]);

      setHealth(healthRes);
      setDatasets(datasetsRes);
      setExperiments(expRes);
      setModels(modelsRes);
    } catch (err) {
      console.error('Failed to load platform data:', err);
      setHealth({ isOnline: false, status: 'offline', compute_backend: 'local' });
    }
  }, []);

  useEffect(() => {
    loadPlatformData();
    const interval = setInterval(loadPlatformData, 15000);
    return () => clearInterval(interval);
  }, [loadPlatformData]);

  const handleLaunchExperiment = (newExp) => {
    setExperiments((prev) => ({
      experiments: [newExp, ...(prev.experiments || [])],
      count: (prev.count || 0) + 1,
    }));
  };

  const handleSelectSection = (sectionId) => {
    setActiveSection(sectionId);
    if (sectionId === 'home') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      const el = document.getElementById(sectionId);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <div className="page-wrapper">
      <Header 
        health={health} 
        activeSection={activeSection}
        onSelectSection={handleSelectSection}
        onOpenLauncher={() => setIsModalOpen(true)}
      />

      <main>
        <Hero 
          onOpenLauncher={() => setIsModalOpen(true)}
          onNavigateToLab={() => handleSelectSection('quantum-lab')}
        />

        <StatsStrip 
          experiments={experiments}
          models={models}
          datasets={datasets}
        />

        <ResearchSections 
          onSelectChapter={(num) => {
            if (num === '02') handleSelectSection('quantum-lab');
            else if (num === '03') handleSelectSection('quantum-lab');
            else if (num === '04') handleSelectSection('reliability');
            else handleSelectSection('experiments');
          }}
        />

        <QuantumLabInteractive 
          onLaunchExperiment={(cfg) => {
            handleLaunchExperiment({
              experiment_id: `QB-${String(Math.floor(Math.random() * 90000) + 10000)}`,
              dataset: 'breast_cancer_diagnostic',
              model_type: `VQC (${cfg.encoding})`,
              qubit_count: cfg.qubits,
              encoding: cfg.encoding,
              noise_model: cfg.noiseModel,
              seed: 42,
              compute_backend: cfg.backend,
              status: 'RUNNING'
            });
            handleSelectSection('experiments');
          }}
        />

        <ReliabilitySection />

        <ComputeInfrastructure />

        <ExperimentsList 
          experiments={experiments.experiments || []} 
          onOpenLauncher={() => setIsModalOpen(true)}
        />

        <ModelRegistrySection 
          models={models.models || []}
        />
      </main>

      <Footer onOpenLauncher={() => setIsModalOpen(true)} />

      <ExperimentModal 
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onLaunch={handleLaunchExperiment}
      />
    </div>
  );
}

export default App;
