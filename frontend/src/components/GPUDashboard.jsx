import React from 'react';

const getHealthBadge = (gpu) => {
  if (gpu?.crashed) {
    return { label: 'CRASHED', cls: 'text-rose-400 bg-rose-500/10 border-rose-500/30' };
  }
  if (gpu?.is_quarantined) {
    return { label: 'QUARANTINED', cls: 'text-purple-400 bg-purple-500/10 border-purple-500/30' };
  }
  if (gpu?.cooldown > 0) {
    return { label: `COOLDOWN (${gpu.cooldown}s)`, cls: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
  }
  const risk = typeof gpu?.risk_score === 'number' ? gpu.risk_score : 0;
  if (risk >= 0.85) {
    return { label: 'CRITICAL', cls: 'text-rose-400 bg-rose-500/10 border-rose-500/30 animate-pulse' };
  }
  if (risk >= 0.75) {
    return { label: 'WARNING', cls: 'text-orange-400 bg-orange-500/10 border-orange-500/30' };
  }
  if ((gpu?.memory ?? 0) > 85 || (gpu?.queue_length ?? 0) > 4) {
    return { label: 'HIGH LOAD', cls: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
  }
  if ((gpu?.running_tasks ?? 0) > 0) {
    return { label: 'ACTIVE', cls: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' };
  }
  return { label: 'HEALTHY', cls: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
};

const getRiskLevel = (score) => {
  if (score === null || score === undefined) return { text: 'N/A', cls: 'text-slate-500' };
  if (score >= 0.85) return { text: 'CRITICAL', cls: 'text-rose-400' };
  if (score >= 0.75) return { text: 'HIGH', cls: 'text-orange-400' };
  if (score >= 0.50) return { text: 'MEDIUM', cls: 'text-amber-400' };
  if (score >= 0.25) return { text: 'LOW', cls: 'text-cyan-400' };
  return { text: 'MINIMAL', cls: 'text-emerald-400' };
};

const tempColor = (t) => {
  if (t === null || t === undefined) return 'text-slate-400';
  if (t >= 80) return 'text-rose-400 font-bold';
  if (t >= 70) return 'text-orange-400 font-semibold';
  if (t >= 60) return 'text-amber-400';
  return 'text-emerald-400';
};

export default function GPUDashboard({
  gpus = [],
  selectedGpuId = 1,
  onSelectGpu,
  lastUpdated,
  isDisconnected = false,
}) {
  if (!gpus || gpus.length === 0) {
    return (
      <div className="glass-panel p-6 text-center space-y-2 border border-white/10">
        <span className="text-3xl">🖥️</span>
        <h4 className="text-sm font-bold text-slate-200">No GPU Telemetry Available</h4>
        <p className="text-xs text-slate-500">
          Waiting for GPU cluster telemetry stream from the backend server...
        </p>
      </div>
    );
  }

  return (
    <section className="space-y-3">
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 px-1">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-base">⚡</span>
          <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
            GPU Cluster Real-Time Health & Partition Matrix
          </h3>
          <span className="text-[10px] bg-slate-900 border border-white/10 text-cyan-400 px-2 py-0.5 rounded-full font-mono">
            {gpus.length} Dedicated Partitions (8 GB each)
          </span>
          <span className="text-[10px] bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 px-2 py-0.5 rounded-full font-mono">
            Simulation Telemetry
          </span>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
          {lastUpdated && (
            <span>
              Updated: <strong className="text-slate-200">{lastUpdated}</strong>
            </span>
          )}
          {isDisconnected && (
            <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse">
              TELEMETRY STALE
            </span>
          )}
        </div>
      </div>

      {/* Responsive Grid of GPU Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {gpus.map((gpu) => {
          const isSelected = selectedGpuId === gpu.id;
          const health = getHealthBadge(gpu);
          const riskInfo = getRiskLevel(gpu.risk_score);
          const capacityGB = gpu.capacity ?? 8.0;
          const memPct = typeof gpu.memory === 'number' ? Math.min(100, Math.max(0, gpu.memory)) : null;
          const allocatedGB = memPct !== null ? ((capacityGB * memPct) / 100).toFixed(1) : 'N/A';
          const freeGB = memPct !== null ? Math.max(0, capacityGB - (capacityGB * memPct) / 100).toFixed(1) : 'N/A';
          const util = typeof gpu.utilization === 'number' ? Math.min(100, Math.max(0, gpu.utilization)) : null;
          const temp = typeof gpu.temperature === 'number' ? gpu.temperature : null;
          const frag = typeof gpu.fragmentation === 'number' ? gpu.fragmentation.toFixed(1) : '0.0';

          return (
            <div
              key={gpu.id}
              onClick={() => onSelectGpu && onSelectGpu(gpu.id)}
              className={`cursor-pointer bg-slate-900/80 border rounded-2xl p-4 space-y-3 transition-all duration-300 backdrop-blur-xl ${
                isSelected
                  ? 'border-cyan-400 ring-2 ring-cyan-500/20 bg-slate-900 shadow-[0_0_25px_rgba(6,182,212,0.18)] scale-[1.01]'
                  : 'border-white/10 hover:border-white/20 hover:bg-slate-900/90'
              }`}
            >
              {/* Card Header */}
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-slate-950 border border-white/10 flex items-center justify-center font-black text-cyan-400 text-xs">
                    G{gpu.id}
                  </div>
                  <div>
                    <h4 className="font-bold text-slate-100 text-sm">GPU {gpu.id}</h4>
                    <span className="text-[10px] text-slate-500 font-mono">ID: partition-0{gpu.id}</span>
                  </div>
                </div>
                <span className={`text-[10px] px-2.5 py-0.5 rounded-full border font-bold font-mono ${health.cls}`}>
                  {health.label}
                </span>
              </div>

              {/* Memory Allocation Bar */}
              <div className="space-y-1 font-mono text-xs">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>VRAM Allocated:</span>
                  <span className="font-bold text-slate-200">
                    {allocatedGB} / {capacityGB.toFixed(1)} GB {memPct !== null ? `(${memPct.toFixed(1)}%)` : ''}
                  </span>
                </div>
                <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      memPct !== null && memPct > 85
                        ? 'bg-gradient-to-r from-orange-500 to-rose-500'
                        : memPct !== null && memPct > 65
                        ? 'bg-gradient-to-r from-cyan-500 to-amber-500'
                        : 'bg-gradient-to-r from-blue-500 to-cyan-400'
                    }`}
                    style={{ width: `${memPct ?? 0}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Free VRAM: {freeGB} GB</span>
                  <span>Frag: {frag}%</span>
                </div>
              </div>

              {/* Utilization Bar */}
              <div className="space-y-1 font-mono text-xs">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Compute Load:</span>
                  <span className="font-bold text-slate-200">
                    {util !== null ? `${util}%` : 'N/A'}
                  </span>
                </div>
                <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-white/5">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-400 transition-all duration-500"
                    style={{ width: `${util ?? 0}%` }}
                  />
                </div>
              </div>

              {/* Hardware Telemetry Grid */}
              <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-xs">
                <div className="bg-slate-950/70 p-1.5 rounded-xl border border-white/5">
                  <span className="text-[9px] text-slate-500 block uppercase">Temp</span>
                  <span className={`text-xs font-bold ${tempColor(temp)}`}>
                    {temp !== null ? `${temp}°C` : 'N/A'}
                  </span>
                </div>
                <div className="bg-slate-950/70 p-1.5 rounded-xl border border-white/5">
                  <span className="text-[9px] text-slate-500 block uppercase">Queue</span>
                  <span className="text-xs font-bold text-slate-200">
                    {gpu.queue_length ?? 0}
                  </span>
                </div>
                <div className="bg-slate-950/70 p-1.5 rounded-xl border border-white/5">
                  <span className="text-[9px] text-slate-500 block uppercase">Running</span>
                  <span className="text-xs font-bold text-cyan-400">
                    {gpu.running_tasks ?? 0}
                  </span>
                </div>
              </div>

              {/* Risk & EFC Footer */}
              <div className="pt-2 border-t border-white/5 flex justify-between items-center text-xs font-mono">
                <div>
                  <span className="text-[10px] text-slate-500 block">RISK SCORE</span>
                  <span className={`font-bold ${riskInfo.cls}`}>
                    {typeof gpu.risk_score === 'number' ? gpu.risk_score.toFixed(4) : 'N/A'}
                  </span>
                  <span className="text-[9px] text-slate-500 ml-1">({riskInfo.text})</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-500 block">EXPECTED FAILURE COST</span>
                  <span className="font-bold text-purple-400">
                    {typeof gpu.efc === 'number' ? gpu.efc.toFixed(4) : '0.0000'}
                  </span>
                </div>
              </div>

              {/* Quarantined or Cooldown notice */}
              {gpu.is_quarantined && (
                <div className="text-[10px] font-mono text-purple-300 bg-purple-500/10 border border-purple-500/30 rounded-lg p-1.5 text-center">
                  🛡️ QUARANTINED: High risk isolation active
                </div>
              )}
              {gpu.cooldown > 0 && (
                <div className="text-[10px] font-mono text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg p-1.5 text-center">
                  ❄️ COOLDOWN: {gpu.cooldown}s remaining
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
