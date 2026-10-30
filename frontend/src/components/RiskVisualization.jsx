import React, { useState } from 'react';
import {
  AreaChart, Area, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';

const getRiskColor = (score) => {
  if (score >= 0.85) return '#ef4444'; // Red
  if (score >= 0.75) return '#f97316'; // Orange
  if (score >= 0.50) return '#eab308'; // Amber
  if (score >= 0.25) return '#22d3ee'; // Cyan
  return '#34d399';                   // Emerald
};

const getRiskLabel = (score) => {
  if (score >= 0.85) return 'CRITICAL';
  if (score >= 0.75) return 'HIGH';
  if (score >= 0.50) return 'MEDIUM';
  if (score >= 0.25) return 'LOW';
  return 'MINIMAL';
};

export default function RiskVisualization({
  gpus = [],
  riskSummary = {},
  riskHistory = [],
  selectedGpuId = 1,
  onSelectGpu,
}) {
  const [activeFactorGpuId, setActiveFactorGpuId] = useState(selectedGpuId || 1);

  const activeGpu = gpus.find((g) => g.id === activeFactorGpuId) || gpus[0] || {};
  const riskComponents = activeGpu.risk_components || {};

  // Dimensions for 8-factor breakdown
  const factorLabels = {
    temperature_risk: { label: 'Temperature Risk', weight: '25%', icon: '🌡️' },
    memory_pressure: { label: 'Memory Pressure', weight: '20%', icon: '💾' },
    fragmentation_risk: { label: 'Fragmentation Risk', weight: '15%', icon: '🧩' },
    utilization_risk: { label: 'Utilization Risk', weight: '10%', icon: '⚡' },
    failure_history: { label: 'Failure History', weight: '15%', icon: '📉' },
    crash_history: { label: 'Crash History', weight: '10%', icon: '💥' },
    anomaly_score: { label: 'Anomaly Score', weight: '5%', icon: '🔍' },
    performance_degradation: { label: 'Degradation', weight: '5%', icon: '⏱️' },
  };

  const radarData = Object.entries(factorLabels).map(([key, meta]) => {
    const rawVal = Number(riskComponents[key]) || 0;
    return {
      dimension: meta.label.split(' ')[0],
      fullName: meta.label,
      value: Math.round(Math.min(1.0, Math.max(0.0, rawVal)) * 100),
      raw: rawVal,
      weight: meta.weight,
    };
  });

  return (
    <section className="glass-panel p-4 space-y-4 border border-white/10">
      {/* Title & Threshold Legend */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-amber-400 text-lg">🛡️</span>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Real-Time Hardware Risk & Failure Vulnerability Engine
            </h3>
            <p className="text-[11px] text-slate-400">
              8-factor risk telemetry, rolling trajectory, and configured fail-safe thresholds
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded-full font-bold">
            Warning: 0.75
          </span>
          <span className="bg-rose-500/10 text-rose-400 border border-rose-500/30 px-2 py-0.5 rounded-full font-bold">
            Quarantine: 0.85
          </span>
        </div>
      </div>

      {/* Trajectory Chart and 8-Factor Breakdown */}
      <div className="grid grid-cols-12 gap-4">
        {/* Rolling Risk & EFC Trajectory Chart */}
        <div className="col-span-12 lg:col-span-7 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-2">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase">
              Cluster Risk & EFC Trajectory
            </span>
            <span className="text-slate-500 text-[10px]">
              Rolling Window: {riskHistory.length || 0} Telemetry Points
            </span>
          </div>

          {riskHistory.length > 0 ? (
            <div className="w-full h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={riskHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="peakGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis dataKey="time_str" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis domain={[0, 1.0]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      return (
                        <div className="bg-slate-900 border border-cyan-500/30 p-2.5 rounded-xl text-xs font-mono shadow-2xl space-y-1">
                          <p className="text-slate-400 border-b border-white/10 pb-1">Time: {label}</p>
                          {payload.map((entry, idx) => (
                            <div key={idx} className="flex justify-between gap-3" style={{ color: entry.color }}>
                              <span>{entry.name}:</span>
                              <span className="font-bold text-white">
                                {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
                              </span>
                            </div>
                          ))}
                        </div>
                      );
                    }}
                  />
                  <Legend verticalAlign="top" height={26} wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />

                  {/* Reference Threshold Lines */}
                  <ReferenceLine
                    y={0.75}
                    stroke="#f59e0b"
                    strokeDasharray="4 4"
                    label={{ value: 'Warning (0.75)', fill: '#f59e0b', fontSize: 10 }}
                  />
                  <ReferenceLine
                    y={0.85}
                    stroke="#ef4444"
                    strokeDasharray="4 4"
                    label={{ value: 'Quarantine (0.85)', fill: '#ef4444', fontSize: 10 }}
                  />

                  <Area
                    type="monotone"
                    name="Cluster Avg Risk"
                    dataKey="cluster_avg_risk"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#riskGrad)"
                  />
                  <Area
                    type="monotone"
                    name="Max GPU Risk"
                    dataKey="max_gpu_risk"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#peakGrad)"
                  />
                  <Line
                    type="monotone"
                    name="Avg EFC"
                    dataKey="avg_efc"
                    stroke="#a78bfa"
                    strokeWidth={2}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-slate-600 text-xs gap-1">
              <span>⏳</span>
              <span>Waiting for initial risk telemetry stream...</span>
            </div>
          )}
        </div>

        {/* 8-Factor Radar & Gauge Breakdown */}
        <div className="col-span-12 lg:col-span-5 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-3">
          {/* GPU Selector for Risk Drill-Down */}
          <div className="flex justify-between items-center">
            <span className="text-[11px] font-bold text-slate-300 uppercase">
              8-Factor Risk Breakdown
            </span>
            <div className="flex gap-1 text-[10px] font-mono">
              {gpus.map((g) => (
                <button
                  key={g.id}
                  onClick={() => {
                    setActiveFactorGpuId(g.id);
                    if (onSelectGpu) onSelectGpu(g.id);
                  }}
                  className={`px-2 py-0.5 rounded-md border font-bold transition-all ${
                    activeFactorGpuId === g.id
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      : 'bg-slate-900 text-slate-400 border-white/5 hover:text-slate-200'
                  }`}
                >
                  G{g.id}
                </button>
              ))}
            </div>
          </div>

          {/* Active GPU Radar Chart */}
          <div className="w-full h-44">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} margin={{ top: 5, right: 25, bottom: 5, left: 25 }}>
                <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar
                  name={`GPU ${activeGpu.id || 1}`}
                  dataKey="value"
                  stroke="#22d3ee"
                  fill="#22d3ee"
                  fillOpacity={0.25}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Factor Component Table / Bars */}
          <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono pt-1 border-t border-white/5">
            {Object.entries(factorLabels).map(([key, meta]) => {
              const val = Number(riskComponents[key]) || 0;
              const pct = Math.round(val * 100);
              return (
                <div key={key} className="bg-slate-900/60 p-1.5 rounded-lg border border-white/5 flex flex-col justify-between">
                  <div className="flex justify-between items-center text-slate-400">
                    <span className="truncate">{meta.icon} {meta.label.split(' ')[0]}</span>
                    <span className="text-[9px] text-slate-500">{meta.weight}</span>
                  </div>
                  <div className="flex justify-between items-center mt-1">
                    <span className="font-bold text-white">{val.toFixed(3)}</span>
                    <span className={pct > 70 ? 'text-rose-400 font-bold' : pct > 40 ? 'text-amber-400' : 'text-emerald-400'}>
                      {pct}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
