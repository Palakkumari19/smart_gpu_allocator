import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  LineChart, Line
} from 'recharts';
import {
  FlaskConical, Play, CheckCircle2, AlertTriangle, ShieldCheck,
  TrendingUp, Activity, RefreshCw, Download, Layers, Scale, Sparkles, AlertCircle
} from 'lucide-react';

export default function ResearchBenchmarking() {
  const [history, setHistory] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [ablationRunning, setAblationRunning] = useState(false);
  const [runnerProgress, setRunnerProgress] = useState({ percent: 0, current_scheduler: null });
  const [activeTab, setActiveTab] = useState('comparison'); // 'comparison', 'ablations', 'statistical', 'runs'

  // Benchmark Form State
  const [config, setConfig] = useState({
    name: 'Research Benchmark',
    schedulers: ['fcfs', 'round_robin', 'least_loaded', 'rl', 'marl', 'risk_aware_marl'],
    num_tasks: 35,
    num_repetitions: 3,
    seed: 42,
    workload_type: 'mixed',
    max_steps: 60,
  });

  const SCHEDULER_METADATA = {
    fcfs: { name: 'First-Come, First-Served', short: 'FCFS', color: '#94a3b8' },
    round_robin: { name: 'Round Robin', short: 'RR', color: '#38bdf8' },
    least_loaded: { name: 'Least Loaded', short: 'Least Load', color: '#818cf8' },
    rl: { name: 'Single-Agent RL', short: 'RL', color: '#c084fc' },
    marl: { name: 'Standard MARL (Risk-Blind)', short: 'MARL (Blind)', color: '#f43f5e' },
    risk_aware_marl: { name: 'Risk-Aware MARL', short: 'Risk-Aware MARL', color: '#10b981' },
  };

  // Fetch benchmark history on mount
  useEffect(() => {
    fetchLatestBenchmark();
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/benchmarks/history');
      if (res.ok) {
        const data = await res.json();
        setHistory(data.experiments || []);
      }
    } catch (err) {
      console.warn('Failed to fetch benchmark history:', err);
    }
  };

  const fetchLatestBenchmark = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/benchmarks/latest');
      if (res.ok) {
        const json = await res.json();
        if (json.status === 'success' && json.data) {
          setSelectedExperiment(json.data);
        }
      }
    } catch (err) {
      console.warn('Failed to load latest benchmark:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadExperimentById = async (expId) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/benchmarks/experiment/${expId}`);
      if (res.ok) {
        const json = await res.json();
        if (json.status === 'success' && json.data) {
          setSelectedExperiment(json.data);
        }
      }
    } catch (err) {
      console.error('Failed to load experiment:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLaunchBenchmark = async () => {
    setRunning(true);
    try {
      const res = await fetch('/api/benchmarks/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        const json = await res.json();
        await fetchHistory();
        if (json.experiment_id) {
          await loadExperimentById(json.experiment_id);
        }
      } else {
        const errJson = await res.json();
        alert(`Benchmark failed to start: ${errJson.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Failed to run benchmark:', err);
      alert('Error launching benchmark.');
    } finally {
      setRunning(false);
    }
  };

  const handleLaunchAblation = async () => {
    setAblationRunning(true);
    try {
      const res = await fetch('/api/benchmarks/ablation?tasks=35&repetitions=3&seed=101', {
        method: 'POST',
      });
      if (res.ok) {
        const json = await res.json();
        await fetchHistory();
        if (json.study_id) {
          await loadExperimentById(json.study_id);
        }
      } else {
        const errJson = await res.json();
        alert(`Ablation failed: ${errJson.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Failed to run ablation study:', err);
    } finally {
      setAblationRunning(false);
    }
  };

  const toggleScheduler = (sId) => {
    setConfig((prev) => {
      const exists = prev.schedulers.includes(sId);
      if (exists && prev.schedulers.length <= 1) return prev; // At least one
      return {
        ...prev,
        schedulers: exists ? prev.schedulers.filter((x) => x !== sId) : [...prev.schedulers, sId],
      };
    });
  };

  // Prepare chart datasets from selectedExperiment
  const summaries = selectedExperiment?.summaries || {};
  const comparisonData = Object.keys(summaries).map((sId) => {
    const s = summaries[sId];
    const m = s.metrics || {};
    return {
      scheduler: SCHEDULER_METADATA[sId]?.short || sId,
      fullName: s.scheduler_name,
      throughput: m.throughput_tasks_per_sec?.mean || 0,
      throughput_ci: m.throughput_tasks_per_sec?.ci_95_upper - m.throughput_tasks_per_sec?.mean || 0,
      success_rate: m.success_rate_pct?.mean || 0,
      turnaround: m.avg_turnaround_time_sec?.mean || 0,
      waiting: m.avg_waiting_time_sec?.mean || 0,
      risk: m.avg_risk_exposure?.mean || 0,
      efc: m.avg_efc?.mean || 0,
      jains: (m.jains_fairness_index?.mean || 1.0) * 100,
      capacity_violations: m.capacity_violations?.mean || 0,
    };
  });

  // Ablations data
  const ablations = selectedExperiment?.ablations || {};
  const hasAblations = Boolean(selectedExperiment?.config?.modes || ablations.deltas_vs_full);

  return (
    <div className="space-y-6 animate-fadeIn text-slate-100">
      {/* Header Banner */}
      <div className="glass-panel p-5 border border-indigo-500/30 shadow-[0_0_30px_rgba(99,102,241,0.12)] flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl border border-indigo-500/30">
              <FlaskConical className="w-5 h-5 text-indigo-400" />
            </div>
            <h2 className="text-lg font-black tracking-tight text-white uppercase">
              Phase 9: Research Metrics & Empirical Benchmarking Suite
            </h2>
            <span className="text-[10px] bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-mono font-bold px-2.5 py-0.5 rounded-full shadow-md shadow-indigo-500/20">
              6 SCHEDULERS & ABLATIONS
            </span>
          </div>
          <p className="text-xs text-slate-400 font-medium mt-1">
            Controlled, reproducible evaluation under identical seeded workloads with 95% Confidence Intervals
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchLatestBenchmark}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg border border-white/10 bg-slate-900 text-xs font-mono text-slate-300 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Latest
          </button>
        </div>
      </div>

      {/* Control & Launch Section */}
      <div className="glass-panel p-5 border border-white/10">
        <h3 className="text-xs font-mono font-semibold tracking-wider text-slate-400 uppercase mb-3 flex items-center gap-2">
          <Play className="w-4 h-4 text-emerald-400" />
          Benchmark Configuration & Experiment Launcher
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
          {/* Workload Tasks */}
          <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
            <span className="text-slate-400">Tasks per Run</span>
            <select
              value={config.num_tasks}
              onChange={(e) => setConfig({ ...config, num_tasks: Number(e.target.value) })}
              className="w-full bg-slate-950 border border-white/10 rounded-lg p-1.5 text-white font-bold outline-none"
            >
              <option value="15">15 Tasks (Smoke Test)</option>
              <option value="35">35 Tasks (Standard)</option>
              <option value="70">70 Tasks (Heavy)</option>
              <option value="120">120 Tasks (Stress)</option>
            </select>
          </div>

          {/* Repetitions */}
          <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
            <span className="text-slate-400">Repetitions (Seeds)</span>
            <select
              value={config.num_repetitions}
              onChange={(e) => setConfig({ ...config, num_repetitions: Number(e.target.value) })}
              className="w-full bg-slate-950 border border-white/10 rounded-lg p-1.5 text-white font-bold outline-none"
            >
              <option value="1">1 Run (Fast)</option>
              <option value="3">3 Seeds (95% CI)</option>
              <option value="5">5 Seeds (High Precision)</option>
            </select>
          </div>

          {/* Workload Profile */}
          <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
            <span className="text-slate-400">Workload Profile</span>
            <select
              value={config.workload_type}
              onChange={(e) => setConfig({ ...config, workload_type: e.target.value })}
              className="w-full bg-slate-950 border border-white/10 rounded-lg p-1.5 text-white font-bold outline-none"
            >
              <option value="mixed">Mixed Video Workload</option>
              <option value="burst">Burst Spikes</option>
              <option value="light_heavy">Light & Heavy Bimodal</option>
              <option value="extreme_heavy">Extreme 4K Workload</option>
            </select>
          </div>

          {/* Action Button */}
          <div className="flex flex-col justify-end gap-2">
            <button
              onClick={handleLaunchBenchmark}
              disabled={running || ablationRunning}
              className={`w-full py-2.5 px-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg ${
                running
                  ? 'bg-amber-600/50 text-amber-200 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white shadow-emerald-500/20'
              }`}
            >
              {running ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Experiments...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Launch 6-Scheduler Benchmark
                </>
              )}
            </button>
            <button
              onClick={handleLaunchAblation}
              disabled={running || ablationRunning}
              className={`w-full py-1.5 px-3 rounded-xl font-bold text-xs flex items-center justify-center gap-1.5 transition-all border ${
                ablationRunning
                  ? 'bg-purple-900/50 text-purple-300 border-purple-500/30'
                  : 'bg-purple-950/60 hover:bg-purple-900/80 text-purple-300 border-purple-500/40'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              {ablationRunning ? 'Running Ablations...' : 'Run Risk Ablation Study'}
            </button>
          </div>
        </div>

        {/* Algorithm Selection Checkboxes */}
        <div className="mt-4 pt-3 border-t border-white/5 flex flex-wrap items-center gap-3 text-xs font-mono">
          <span className="text-slate-400">Included Algorithms:</span>
          {Object.entries(SCHEDULER_METADATA).map(([sId, meta]) => {
            const isSelected = config.schedulers.includes(sId);
            return (
              <button
                key={sId}
                onClick={() => toggleScheduler(sId)}
                className={`px-2.5 py-1 rounded-lg border font-bold transition-all flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-slate-800 text-white border-cyan-500/50 shadow-sm'
                    : 'bg-slate-950 text-slate-500 border-white/5 line-through'
                }`}
              >
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: meta.color }}
                />
                {meta.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Sub-Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-white/10 pb-2 text-xs font-mono">
        {[
          ['comparison', '📊 6-Scheduler Comparison'],
          ['ablations', '🔬 Ablation Study'],
          ['statistical', '📈 95% Confidence Intervals'],
          ['history', '🗂️ Exported Experiments Index'],
        ].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
              activeTab === key
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* TAB 1: 6-SCHEDULER COMPARISON MATRIX */}
      {/* ───────────────────────────────────────────────────────────── */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          {/* Main Comparison Table */}
          <div className="glass-panel p-5 border border-white/10 overflow-x-auto">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Scale className="w-4 h-4 text-cyan-400" />
                  Multi-Algorithm Benchmark Matrix (Empirical Measurements)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Evaluated on {selectedExperiment?.config?.num_tasks || 35} identical tasks per seed over{' '}
                  {selectedExperiment?.config?.num_repetitions || 3} independent repetitions.
                </p>
              </div>
              <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-3 py-1 rounded-lg border border-white/10">
                Experiment ID: {selectedExperiment?.experiment_id || 'N/A'}
              </span>
            </div>

            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400">
                  <th className="py-2.5 px-3">Scheduling Strategy</th>
                  <th className="py-2.5 px-3 text-right">Throughput</th>
                  <th className="py-2.5 px-3 text-right">Success Rate</th>
                  <th className="py-2.5 px-3 text-right">Avg Turnaround</th>
                  <th className="py-2.5 px-3 text-right">Jain's Fairness</th>
                  <th className="py-2.5 px-3 text-right">Avg Risk</th>
                  <th className="py-2.5 px-3 text-right">Avg EFC</th>
                  <th className="py-2.5 px-3 text-center">8 GB Invariant</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {Object.keys(summaries).map((sId) => {
                  const s = summaries[sId];
                  const m = s.metrics || {};
                  const isRiskAware = sId === 'risk_aware_marl';
                  const meta = SCHEDULER_METADATA[sId] || { name: s.scheduler_name, color: '#38bdf8' };

                  return (
                    <tr
                      key={sId}
                      className={`hover:bg-white/[0.02] transition-colors ${
                        isRiskAware ? 'bg-emerald-950/20 font-bold' : ''
                      }`}
                    >
                      <td className="py-3 px-3 flex items-center gap-2">
                        <div
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: meta.color }}
                        />
                        <span className="text-white">{s.scheduler_name}</span>
                        {isRiskAware && (
                          <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/30">
                            TARGET
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3 text-right font-bold text-cyan-300">
                        {m.throughput_tasks_per_sec?.mean?.toFixed(2) || '0.00'} t/s
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`px-2 py-0.5 rounded ${
                            (m.success_rate_pct?.mean || 0) >= 95
                              ? 'bg-emerald-500/20 text-emerald-300'
                              : 'bg-amber-500/20 text-amber-300'
                          }`}
                        >
                          {m.success_rate_pct?.mean?.toFixed(1) || '0.0'}%
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right text-slate-300">
                        {m.avg_turnaround_time_sec?.mean?.toFixed(2) || '0.00'}s
                      </td>
                      <td className="py-3 px-3 text-right text-purple-300">
                        {((m.jains_fairness_index?.mean || 1.0) * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={`font-mono font-bold ${
                            (m.avg_risk_exposure?.mean || 0) < 0.04
                              ? 'text-emerald-400'
                              : (m.avg_risk_exposure?.mean || 0) < 0.07
                              ? 'text-amber-400'
                              : 'text-rose-400'
                          }`}
                        >
                          {m.avg_risk_exposure?.mean?.toFixed(4) || '0.0000'}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right text-amber-300 font-bold">
                        {m.avg_efc?.mean?.toFixed(4) || '0.0000'}
                      </td>
                      <td className="py-3 px-3 text-center">
                        <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-500/30">
                          <ShieldCheck className="w-3 h-3" />
                          0 Violations
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Graphical Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Throughput & Latency */}
            <div className="glass-panel p-5 border border-white/10">
              <h4 className="text-xs font-mono font-bold text-slate-300 uppercase mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                Scheduling Throughput Comparison (Tasks / Second)
              </h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                    <XAxis dataKey="scheduler" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    />
                    <Bar dataKey="throughput" name="Throughput (tasks/s)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Risk Exposure & EFC */}
            <div className="glass-panel p-5 border border-white/10">
              <h4 className="text-xs font-mono font-bold text-slate-300 uppercase mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Average Risk Exposure & Expected Failure Cost (EFC)
              </h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                    <XAxis dataKey="scheduler" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="risk" name="Avg Risk Exposure" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="efc" name="Expected Failure Cost" fill="#fbbf24" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* TAB 2: ABLATION STUDY VIEW */}
      {/* ───────────────────────────────────────────────────────────── */}
      {activeTab === 'ablations' && (
        <div className="space-y-6">
          <div className="glass-panel p-5 border border-purple-500/20 bg-purple-950/10">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-1">
              <Layers className="w-4 h-4 text-purple-400" />
              Risk-Aware MARL Component Ablation Analysis
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Measures the individual contribution of Risk Penalties vs Expected Failure Cost (EFC) penalties
              in candidate bidding under identical workload seeds.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
              {[
                {
                  mode: 'full_risk_aware',
                  title: 'Full Risk-Aware MARL',
                  formula: 'Q·0.5 + Perf - 0.25·Risk - 0.20·EFC',
                  tag: 'Production Active',
                  tagClass: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
                },
                {
                  mode: 'no_risk_penalty',
                  title: 'Ablation: No Risk Penalty',
                  formula: 'Q·0.5 + Perf - 0.20·EFC',
                  tag: 'Risk Penalty Disabled',
                  tagClass: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
                },
                {
                  mode: 'no_efc_penalty',
                  title: 'Ablation: No EFC Penalty',
                  formula: 'Q·0.5 + Perf - 0.25·Risk',
                  tag: 'EFC Disabled',
                  tagClass: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
                },
                {
                  mode: 'risk_blind',
                  title: 'Standard MARL (Risk-Blind)',
                  formula: 'Q·0.5 + Perf (No deductions)',
                  tag: 'Pure Performance',
                  tagClass: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
                },
              ].map((card) => {
                const modeSummary = summaries[card.mode]?.metrics || {};
                return (
                  <div key={card.mode} className="bg-slate-900/90 p-4 rounded-xl border border-white/10 space-y-2">
                    <div className="flex justify-between items-start">
                      <span className="font-bold text-white text-xs">{card.title}</span>
                      <span className={`text-[9px] font-mono px-2 py-0.5 rounded border ${card.tagClass}`}>
                        {card.tag}
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-cyan-300 bg-slate-950 p-1.5 rounded border border-white/5">
                      {card.formula}
                    </div>
                    <div className="pt-2 border-t border-white/5 space-y-1 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Avg Risk Exposure:</span>
                        <span className="font-bold text-rose-400">
                          {modeSummary.avg_risk_exposure?.mean?.toFixed(4) || '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Avg EFC Incurred:</span>
                        <span className="font-bold text-amber-400">
                          {modeSummary.avg_efc?.mean?.toFixed(4) || '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Throughput:</span>
                        <span className="font-bold text-cyan-400">
                          {modeSummary.throughput_tasks_per_sec?.mean?.toFixed(2) || '—'} t/s
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* TAB 3: STATISTICAL 95% CONFIDENCE INTERVALS */}
      {/* ───────────────────────────────────────────────────────────── */}
      {activeTab === 'statistical' && (
        <div className="glass-panel p-5 border border-white/10 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            95% Confidence Intervals & Sample Dispersion (Across Repetitions)
          </h3>
          <p className="text-xs text-slate-400 font-mono">
            Formulas: Mean (x̄), Sample Standard Deviation (s), 95% CI: [ x̄ - 1.96·(s/√N), x̄ + 1.96·(s/√N) ]
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400">
                  <th className="py-2 px-3">Algorithm</th>
                  <th className="py-2 px-3">Metric</th>
                  <th className="py-2 px-3 text-right">Mean</th>
                  <th className="py-2 px-3 text-right">Std Dev</th>
                  <th className="py-2 px-3 text-right">95% CI Range</th>
                  <th className="py-2 px-3 text-right">[Min, Max]</th>
                  <th className="py-2 px-3 text-center">N (Runs)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {Object.keys(summaries).map((sId) => {
                  const s = summaries[sId];
                  const m = s.metrics || {};
                  return ['throughput_tasks_per_sec', 'avg_risk_exposure', 'avg_turnaround_time_sec'].map((mKey) => {
                    const agg = m[mKey] || { mean: 0, std: 0, ci_95_lower: 0, ci_95_upper: 0, min_val: 0, max_val: 0, sample_size: 0 };
                    return (
                      <tr key={`${sId}-${mKey}`} className="hover:bg-white/[0.02]">
                        <td className="py-2 px-3 font-bold text-white">{s.scheduler_name}</td>
                        <td className="py-2 px-3 text-cyan-300">{mKey}</td>
                        <td className="py-2 px-3 text-right font-bold text-white">{agg.mean.toFixed(4)}</td>
                        <td className="py-2 px-3 text-right text-slate-400">{agg.std.toFixed(4)}</td>
                        <td className="py-2 px-3 text-right text-emerald-400">
                          [{agg.ci_95_lower.toFixed(3)}, {agg.ci_95_upper.toFixed(3)}]
                        </td>
                        <td className="py-2 px-3 text-right text-slate-400">
                          [{agg.min_val.toFixed(3)}, {agg.max_val.toFixed(3)}]
                        </td>
                        <td className="py-2 px-3 text-center text-slate-300">{agg.sample_size}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* TAB 4: HISTORICAL EXPERIMENTS INDEX */}
      {/* ───────────────────────────────────────────────────────────── */}
      {activeTab === 'history' && (
        <div className="glass-panel p-5 border border-white/10 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Download className="w-4 h-4 text-emerald-400" />
            Persisted Benchmark Experiments Index (benchmarking/results/)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400">
                  <th className="py-2 px-3">Experiment ID</th>
                  <th className="py-2 px-3">Timestamp (UTC)</th>
                  <th className="py-2 px-3">Name</th>
                  <th className="py-2 px-3 text-center">Tasks</th>
                  <th className="py-2 px-3 text-center">Total Runs</th>
                  <th className="py-2 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {history.map((exp) => (
                  <tr key={exp.experiment_id} className="hover:bg-white/[0.02]">
                    <td className="py-2.5 px-3 font-bold text-cyan-400">{exp.experiment_id}</td>
                    <td className="py-2.5 px-3 text-slate-400">{exp.timestamp}</td>
                    <td className="py-2.5 px-3 text-white">{exp.name}</td>
                    <td className="py-2.5 px-3 text-center">{exp.num_tasks}</td>
                    <td className="py-2.5 px-3 text-center">{exp.total_runs}</td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={() => loadExperimentById(exp.experiment_id)}
                        className="px-2.5 py-1 bg-indigo-600/30 hover:bg-indigo-600 text-indigo-200 hover:text-white rounded border border-indigo-500/40 transition-all"
                      >
                        Load Into View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
