import React from 'react';

export function ComputeInfrastructure() {
  return (
    <section className="infra-section">
      <div className="editorial-container">
        <div className="eyebrow">Hardware Architecture</div>
        <h2 className="section-intro-title">High-performance simulation and edge telemetry.</h2>

        <div className="infra-grid">
          {/* DGX B200 */}
          <div className="infra-card">
            <div className="infra-chip">HPC Acceleration</div>
            <h3 className="infra-name">NVIDIA DGX B200</h3>
            <p className="infra-desc">
              Dedicated classical GPU computing cluster powering parallelized PyTorch classical model training, cuQuantum statevector/tensor-network circuit simulation, and high-throughput experiment parameter sweeps.
            </p>
            <div className="infra-notice">
              Notice: NVIDIA DGX B200 is a classical high-performance compute system used for model training and quantum circuit simulation. It is not a quantum computer.
            </div>
          </div>

          {/* Raspberry Pi */}
          <div className="infra-card">
            <div className="infra-chip">Edge Research Terminal</div>
            <h3 className="infra-name">Raspberry Pi + 7″ Display</h3>
            <p className="infra-desc">
              Lightweight edge terminal interface deployed in laboratory settings for displaying live experiment telemetry, monitoring remote campaign execution, and presenting calibrated inference visualizers.
            </p>
            <div className="infra-notice">
              Notice: Raspberry Pi is an edge visualization and research client. It is not a quantum processor.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
