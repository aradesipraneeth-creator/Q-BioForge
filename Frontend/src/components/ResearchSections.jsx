import React from 'react';

const CHAPTERS = [
  {
    num: '01',
    title: 'Data Laboratory',
    desc: 'Leakage-safe preprocessing of biomedical tabular records. Stratified partitioning isolates training distributions before applying imputation, outlier filters, and feature transformations bounded into quantum-ready angular ranges.',
    tags: ['Wisconsin Breast Cancer', 'Heart Disease', 'Diabetes Tabular', 'Zero Data Leakage', 'Bounded Encodings'],
    action: 'Inspect Pipeline'
  },
  {
    num: '02',
    title: 'Quantum Laboratory',
    desc: 'Configurable variational quantum classifiers (VQC), quantum support vector machines (QSVM), and hybrid QNN neural layers. Hardware-agnostic ansatz parameterization across Pennylane and Qiskit simulator runtimes.',
    tags: ['Angle Encoding', 'Amplitude Encoding', 'Data Re-uploading', 'Parameterized Ansatz', 'cuQuantum Ready'],
    action: 'Open Circuit Lab'
  },
  {
    num: '03',
    title: 'NISQ Noise Lab',
    desc: 'Systematic injection of realistic quantum hardware error channels. Profiles decision boundary degradation under readout infidelity, single/two-qubit depolarizing noise, and T1/T2 thermal relaxation.',
    tags: ['Readout Errors', 'Depolarizing Channel', 'T1 / T2 Relaxation', 'Pauli Bit-Flip', 'Robustness Curves'],
    action: 'Configure Noise'
  },
  {
    num: '04',
    title: 'Reliability Engine',
    desc: 'Tri-state clinical decision support framework. Combines Brier calibration scores, deep model disagreement, and distribution shift detection to govern selective prediction and safe abstention.',
    tags: ['ACCEPT / REVIEW / ABSTAIN', 'Brier Calibration', 'Uncertainty Bounds', 'Distribution Shift', 'Decision Support'],
    action: 'View Triad Model'
  },
];

export function ResearchSections({ onSelectChapter }) {
  return (
    <section className="editorial-sections" id="research">
      <div className="editorial-container">
        <div className="section-intro-header">
          <div className="eyebrow">Platform Architecture</div>
          <h2 className="section-intro-title">Four pillars of noise-aware quantum biomedical ML.</h2>
        </div>

        <div className="research-chapters-list">
          {CHAPTERS.map((ch) => (
            <div 
              key={ch.num} 
              className="chapter-row"
              onClick={() => onSelectChapter(ch.num)}
            >
              <div className="chapter-num">{ch.num}</div>
              <div className="chapter-title">{ch.title}</div>
              <div>
                <p className="chapter-details">{ch.desc}</p>
                <div className="chapter-tags">
                  {ch.tags.map((t) => (
                    <span key={t} className="tag-pill">{t}</span>
                  ))}
                </div>
              </div>
              <div className="chapter-action">
                {ch.action} →
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
