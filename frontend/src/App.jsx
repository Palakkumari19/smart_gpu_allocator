import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, Cell
} from 'recharts';
import wsService from './services/websocket';
import Bit8Dashboard from './components/Bit8Dashboard';
import ResearchBenchmarking from './components/ResearchBenchmarking';

// ─── API BASE URL ────────────────────────────────────────────────────────────
const getApiUrl = () => {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  const protocol = window.location.protocol;
  const host = window.location.hostname || 'localhost';
  const port = window.location.port !== '8000' ? '8000' : window.location.port;
  return `${protocol}//${host}:${port}`;
};
const API = getApiUrl();

// ─── COLOR & BADGE HELPERS ──────────────────────────────────────────────────
const GPU_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#fb923c'];

const getRiskColor = (score) => {
  if (score >= 0.85) return '#ef4444'; // Red (Quarantine)
  if (score >= 0.75) return '#f97316'; // Orange (Warning)
  if (score >= 0.50) return '#eab308'; // Amber (Medium)
  if (score >= 0.25) return '#22d3ee'; // Cyan (Low)
  return '#34d399';                   // Emerald (Minimal)
};

const getRiskLabel = (score) => {
  if (score >= 0.85) return 'CRITICAL';
  if (score >= 0.75) return 'HIGH';
  if (score >= 0.50) return 'MEDIUM';
  if (score >= 0.25) return 'LOW';
  return 'MINIMAL';
};

const getHealthStatus = (gpu) => {
  if (gpu.crashed) return { label: 'CRASHED', cls: 'text-red-400 bg-red-500/10 border-red-500/30' };
  if (gpu.is_quarantined) return { label: 'QUARANTINED', cls: 'text-purple-400 bg-purple-500/10 border-purple-500/30' };
  if (gpu.cooldown > 0) return { label: `COOLDOWN (${gpu.cooldown}s)`, cls: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
  if ((gpu.risk_score ?? 0) >= 0.75) return { label: 'WARNING', cls: 'text-orange-400 bg-orange-500/10 border-orange-500/30' };
  if ((gpu.memory ?? 0) > 80 || (gpu.queue_length ?? 0) > 3) return { label: 'HIGH LOAD', cls: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
  if ((gpu.running_tasks ?? 0) > 0) return { label: 'ACTIVE', cls: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' };
  return { label: 'HEALTHY', cls: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
};

const severityColor = {
  LOW: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  MEDIUM: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
  HIGH: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  'VERY HIGH': 'text-red-400 bg-red-500/10 border-red-500/30',
  CRITICAL: 'text-rose-400 bg-rose-500/10 border-rose-500/30',
};

const tempColor = (t) => (t >= 80 ? 'text-red-400' : t >= 70 ? 'text-orange-400' : t >= 60 ? 'text-amber-400' : 'text-emerald-400');

// ─── SUB-COMPONENTS ──────────────────────────────────────────────────────────

const MetricCard = ({ label, value, sub, accent = 'cyan', icon, alertBadge, denominator }) => {
  const borderAccents = {
    cyan: 'border-cyan-500/30 hover:border-cyan-500/60 shadow-[0_0_20px_rgba(34,211,238,0.06)]',
    amber: 'border-amber-500/30 hover:border-amber-500/60 shadow-[0_0_20px_rgba(245,158,11,0.06)]',
    rose: 'border-rose-500/30 hover:border-rose-500/60 shadow-[0_0_20px_rgba(244,63,94,0.06)]',
    emerald: 'border-emerald-500/30 hover:border-emerald-500/60 shadow-[0_0_20px_rgba(52,211,153,0.06)]',
    purple: 'border-purple-500/30 hover:border-purple-500/60 shadow-[0_0_20px_rgba(167,139,250,0.06)]',
  };
  const textAccents = {
    cyan: 'text-cyan-400',
    amber: 'text-amber-400',
    rose: 'text-rose-400',
    emerald: 'text-emerald-400',
    purple: 'text-purple-400',
  };

  return (
    <div className={`relative bg-slate-900/60 border rounded-2xl p-4 flex flex-col justify-between backdrop-blur-xl transition-all duration-300 ${borderAccents[accent] || borderAccents.cyan}`}>
      <div className="flex justify-between items-start mb-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
        <div className="flex items-center gap-1.5">
          {alertBadge && (
            <span className="text-[9px] px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse font-bold">
              {alertBadge}
            </span>
          )}
          {icon && <span className="text-base opacity-80">{icon}</span>}
        </div>
      </div>
      <div className="my-1">
        <span className={`text-2xl lg:text-3xl font-black tabular-nums tracking-tight ${textAccents[accent] || textAccents.cyan}`}>
          {value}
        </span>
      </div>
      <div className="flex flex-col gap-0.5 mt-1 border-t border-white/5 pt-1.5">
        {sub && <span className="text-[11px] text-slate-400 font-medium">{sub}</span>}
        {denominator && <span className="text-[9px] text-slate-500 italic">{denominator}</span>}
      </div>
    </div>
  );
};

const CustomChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-950/95 border border-cyan-500/30 rounded-xl p-3 text-xs shadow-2xl backdrop-blur-xl space-y-1">
      <div className="flex items-center justify-between gap-4 border-b border-white/10 pb-1 text-slate-400 font-mono text-[11px]">
        <span>Time: {label}</span>
      </div>
      {payload.map((entry, idx) => (
        <div key={idx} className="flex items-center justify-between gap-4 text-[11px]">
          <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            {entry.name}:
          </span>
          <span className="font-mono font-bold text-slate-100">
            {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
};

const CustomDecisionTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;
  return (
    <div className="bg-slate-950/95 border border-emerald-500/30 rounded-xl p-3 text-xs shadow-2xl backdrop-blur-xl space-y-1 min-w-[200px]">
      <div className="font-bold text-slate-200 border-b border-white/10 pb-1 flex justify-between">
        <span>{data.label}</span>
        <span className={data.eligible ? 'text-emerald-400 font-mono' : 'text-rose-400 font-mono'}>
          {data.eligible ? 'ELIGIBLE' : data.ineligible_reason || 'INELIGIBLE'}
        </span>
      </div>
      <div className="space-y-0.5 text-[11px] pt-1 font-mono">
        <div className="flex justify-between text-slate-300">
          <span>Bid Score:</span>
          <span className="font-bold text-cyan-400">{data.bid_score > -50 ? data.bid_score.toFixed(4) : 'N/A'}</span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Risk Penalty:</span>
          <span className="text-rose-400">-{data.risk_penalty?.toFixed(4) ?? '0.0000'}</span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>EFC Penalty:</span>
          <span className="text-amber-400">-{data.efc_penalty?.toFixed(4) ?? '0.0000'}</span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Perf Score:</span>
          <span className="text-emerald-400">+{data.perf_score?.toFixed(4) ?? '0.0000'}</span>
        </div>
        <div className="flex justify-between text-slate-400">
          <span>Q Contribution:</span>
          <span className="text-purple-400">+{data.q_contribution?.toFixed(4) ?? '0.0000'}</span>
        </div>
      </div>
    </div>
  );
};

// ─── RISK RADAR CHART COMPONENT ──────────────────────────────────────────────
function RiskRadarChart({ components, gpuLabel }) {
  if (!components || Object.keys(components).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-slate-600 text-xs gap-1">
        <span>📊</span>
        <span>No risk telemetry yet. Start simulation to inspect live risk factors.</span>
      </div>
    );
  }

  const dimNames = {
    temperature_risk: 'Temp Risk',
    memory_pressure: 'Mem Press',
    fragmentation_risk: 'Frag Risk',
    utilization_risk: 'Util Risk',
    failure_history: 'Fail Hist',
    crash_history: 'Crash Hist',
    anomaly_score: 'Anomaly',
    performance_degradation: 'Degradation',
  };

  const chartData = Object.entries(components).map(([k, v]) => ({
    dimension: dimNames[k] || k,
    value: Math.round(Math.min(1.0, Math.max(0.0, Number(v) || 0)) * 100),
    raw: Number(v) || 0,
  }));

  return (
    <div className="w-full h-52">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={chartData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
          <PolarGrid stroke="#334155" strokeDasharray="3 3" />
          <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 500 }} />
          <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
          <Radar name={gpuLabel || 'Selected GPU'} dataKey="value" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.25} />
          <Tooltip content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const item = payload[0]?.payload;
            return (
              <div className="bg-slate-950/95 border border-cyan-500/40 rounded-lg p-2 text-xs shadow-xl font-mono">
                <span className="text-cyan-400 font-bold">{item.dimension}: </span>
                <span className="text-white">{item.raw.toFixed(4)} ({item.value}%)</span>
              </div>
            );
          }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── MAIN APP COMPONENT ──────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('bit8'); // 'bit8' | 'monitor' | 'workload' | 'unified'
  const [tasksList, setTasksList] = useState([]);
  const [lastUpdatedTime, setLastUpdatedTime] = useState(() => new Date().toLocaleTimeString());
  const [scheduler, setScheduler] = useState('marl');
  const [speed, setSpeed] = useState('Medium');
  const [systemStatus, setSystemStatus] = useState('Disconnected');
  const [isRunning, setIsRunning] = useState(false);

  // Selected GPU for risk breakdown drill-down
  const [selectedGpuId, setSelectedGpuId] = useState(1);

  // Core cluster telemetry
  const [gpus, setGpus] = useState([
    { id: 1, memory: 0, temperature: 30, utilization: 0, queue_length: 0, running_tasks: 0, completed_tasks: 0, fragmentation: 0, crashed: false, cooldown: 0, risk_score: 0, efc: 0 },
    { id: 2, memory: 0, temperature: 30, utilization: 0, queue_length: 0, running_tasks: 0, completed_tasks: 0, fragmentation: 0, crashed: false, cooldown: 0, risk_score: 0, efc: 0 },
    { id: 3, memory: 0, temperature: 30, utilization: 0, queue_length: 0, running_tasks: 0, completed_tasks: 0, fragmentation: 0, crashed: false, cooldown: 0, risk_score: 0, efc: 0 },
    { id: 4, memory: 0, temperature: 30, utilization: 0, queue_length: 0, running_tasks: 0, completed_tasks: 0, fragmentation: 0, crashed: false, cooldown: 0, risk_score: 0, efc: 0 },
  ]);

  const [metrics, setMetrics] = useState({});
  const [logs, setLogs] = useState([{ id: 'init-0', time: '00:00:00', type: 'info', message: 'AetherGrid Risk Intelligence connected' }]);
  const [videoQueue, setVideoQueue] = useState([]);
  const [marlDecision, setMarlDecision] = useState(null);
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [riskHistory, setRiskHistory] = useState([]);
  const [riskSummary, setRiskSummary] = useState({
    cluster_avg_risk: 0.0,
    max_gpu_risk: 0.0,
    max_risk_gpu_id: 1,
    avg_efc: 0.0,
    risk_affected_decisions: 0,
    quarantined_count: 0,
    cooldown_count: 0,
    scheduling_success_rate: 100.0,
    threshold_warning: 0.75,
    threshold_quarantine: 0.85,
  });

  // Video workload states
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [currentVideoMeta, setCurrentVideoMeta] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  // Filter for event timeline
  const [eventFilter, setEventFilter] = useState('all'); // 'all' | 'risk' | 'error' | 'decision'

  // Fetch initial REST snapshot for zero-latency dashboard hydration
  const fetchInitialSnapshots = useCallback(async () => {
    try {
      const [resMon, resDec, resEvt, resTasks] = await Promise.all([
        fetch(`${API}/api/risk/monitoring`).then(r => r.json()).catch(() => null),
        fetch(`${API}/api/scheduler/decisions?limit=20`).then(r => r.json()).catch(() => null),
        fetch(`${API}/api/events?limit=40`).then(r => r.json()).catch(() => null),
        fetch(`${API}/api/tasks?limit=50`).then(r => r.json()).catch(() => null),
      ]);

      if (resTasks && Array.isArray(resTasks.tasks)) {
        setTasksList(resTasks.tasks);
      }

      if (resMon && resMon.monitoring) {
        if (resMon.monitoring.summary) setRiskSummary(resMon.monitoring.summary);
        if (resMon.monitoring.history?.length) setRiskHistory(resMon.monitoring.history);
        if (resMon.monitoring.latest_decision) setMarlDecision(resMon.monitoring.latest_decision);
        if (resMon.gpus?.length) setGpus(resMon.gpus);
      }
      if (resDec && resDec.decisions?.length) {
        setDecisionHistory(resDec.decisions);
      }
      if (resEvt && resEvt.events?.length) {
        setLogs(resEvt.events);
      }
    } catch (err) {
      console.warn('Initial snapshot fetch error:', err);
    }
  }, []);

  // ── WebSocket Connection & Event Handler ───────────────────────────────────
  useEffect(() => {
    fetchInitialSnapshots();

    wsService.connect((data, raw) => {
      setLastUpdatedTime(new Date().toLocaleTimeString());

      // 0. Real-time task state updates
      if (data.tasks && Array.isArray(data.tasks)) {
        setTasksList((prev) => {
          const taskMap = new Map();
          // Keep existing tasks
          prev.forEach((t) => {
            const id = t.task_id || t.id;
            if (id) taskMap.set(id, t);
          });
          // Merge newer snapshot updates
          data.tasks.forEach((t) => {
            const id = t.task_id || t.id;
            if (id) {
              const existing = taskMap.get(id) || {};
              taskMap.set(id, { ...existing, ...t });
            }
          });
          return Array.from(taskMap.values()).slice(0, 100);
        });
      } else if (data.task_registry && typeof data.task_registry === 'object') {
        // Fallback from environment task registry
        const regTasks = Object.entries(data.task_registry).map(([tid, meta]) => ({
          task_id: tid,
          status: (meta.status || 'QUEUED').toUpperCase(),
          requested_memory_gb: meta.memory_required || 2.0,
          assigned_gpu: meta.assigned_gpu,
          submission_time: meta.arrival_time ? new Date(meta.arrival_time * 1000).toISOString() : null,
          start_time: meta.start_time ? new Date(meta.start_time * 1000).toISOString() : null,
          completion_time: meta.completion_time ? new Date(meta.completion_time * 1000).toISOString() : null,
        }));
        if (regTasks.length > 0) {
          setTasksList((prev) => {
            const taskMap = new Map();
            prev.forEach((t) => taskMap.set(t.task_id || t.id, t));
            regTasks.forEach((t) => {
              const existing = taskMap.get(t.task_id) || {};
              taskMap.set(t.task_id, { ...existing, ...t });
            });
            return Array.from(taskMap.values()).slice(0, 100);
          });
        }
      }

      // 1. GPU state updates
      if (data.gpus) {
        setGpus(data.gpus.map(g => ({ ...g, memory: g.memory ?? 0 })));
      }
      // 2. Metrics & Execution status
      if (data.metrics) setMetrics(data.metrics);
      if (data.running !== undefined) setIsRunning(data.running);
      if (data.video_queue) setVideoQueue(data.video_queue);

      // 3. Events with deduplication
      if (data.events && Array.isArray(data.events)) {
        setLogs(prev => {
          const combined = [...data.events, ...prev];
          const seen = new Set();
          const deduped = [];
          for (const item of combined) {
            const key = item.id || `${item.time}-${item.message}`;
            if (!seen.has(key)) {
              seen.add(key);
              deduped.push(item);
            }
          }
          return deduped.slice(0, 80);
        });
      }

      // 4. MARL Decision updates
      if (data.last_marl_decision && Object.keys(data.last_marl_decision).length > 0) {
        setMarlDecision(data.last_marl_decision);
        setDecisionHistory(prev => {
          const exists = prev.some(d => d.decision_id === data.last_marl_decision.decision_id);
          if (!exists) {
            return [data.last_marl_decision, ...prev].slice(0, 30);
          }
          return prev;
        });
      }

      // 5. Video Meta
      if (data.last_video_meta && Object.keys(data.last_video_meta).length > 0) {
        setCurrentVideoMeta(data.last_video_meta);
      }

      // 6. Risk Monitoring Snapshot & History
      if (data.risk_monitoring) {
        if (data.risk_monitoring.summary) {
          setRiskSummary(data.risk_monitoring.summary);
        }
        if (data.risk_monitoring.history && Array.isArray(data.risk_monitoring.history)) {
          setRiskHistory(data.risk_monitoring.history);
        }
        if (data.risk_monitoring.recent_decisions) {
          setDecisionHistory(data.risk_monitoring.recent_decisions);
        }
      }

      // 7. Typed WebSocket Event handling
      if (raw?.type === 'task.lifecycle' && raw.data) {
        const item = raw.data;
        const tid = item.task_id || item.id;
        if (tid) {
          setTasksList((prev) => {
            const idx = prev.findIndex((t) => (t.task_id || t.id) === tid);
            if (idx >= 0) {
              const copy = [...prev];
              copy[idx] = { ...copy[idx], ...item };
              return copy;
            } else {
              return [item, ...prev].slice(0, 100);
            }
          });
        }
      }
      if (raw?.type === 'risk.updated' && raw.data) {
        setRiskSummary(prev => ({ ...prev, ...raw.data }));
      }
      if (raw?.type === 'scheduler.decision' && raw.data) {
        setMarlDecision(raw.data);
        setDecisionHistory(prev => [raw.data, ...prev.filter(d => d.decision_id !== raw.data.decision_id)].slice(0, 30));
      }
      if (raw?.type === 'event.created' && raw.data) {
        setLogs(prev => [raw.data, ...prev].slice(0, 80));
      }
    }, setSystemStatus);

    // Periodic safety sync (every 5 seconds) to ensure fresh snapshot
    const syncInterval = setInterval(() => {
      fetch(`${API}/api/tasks?limit=50`)
        .then(r => r.json())
        .then(res => {
          if (res?.tasks && Array.isArray(res.tasks)) {
            setTasksList(prev => {
              const taskMap = new Map();
              prev.forEach(t => taskMap.set(t.task_id || t.id, t));
              res.tasks.forEach(t => {
                const id = t.task_id || t.id;
                if (id) {
                  const existing = taskMap.get(id) || {};
                  taskMap.set(id, { ...existing, ...t });
                }
              });
              return Array.from(taskMap.values()).slice(0, 100);
            });
          }
        })
        .catch(() => {});
    }, 5000);

    return () => {
      clearInterval(syncInterval);
      wsService.disconnect();
    };
  }, [fetchInitialSnapshots]);

  // ── Control Helpers ────────────────────────────────────────────────────────
  const post = async (endpoint, body = {}) => {
    try {
      await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err) {
      console.error('API Error:', err);
    }
  };

  const applyScheduler = (newSched) => {
    const s = newSched || scheduler;
    const map = {
      MARL: 'marl',
      marl: 'marl',
      'Single RL': 'rl',
      rl: 'rl',
      FCFS: 'fcfs',
      fcfs: 'fcfs',
      'Round Robin': 'round_robin',
      round_robin: 'round_robin',
      'Least Loaded': 'least_loaded',
      least_loaded: 'least_loaded',
      Random: 'random',
      random: 'random',
    };
    post('/set-scheduler', { scheduler: map[s] || 'marl' });
  };

  // ── Drag & Drop Video Handling ─────────────────────────────────────────────
  const onDragOver = useCallback((e) => {
    e.preventDefault();
    dropZoneRef.current?.classList.add('border-cyan-500');
  }, []);
  const onDragLeave = useCallback(() => dropZoneRef.current?.classList.remove('border-cyan-500'), []);
  const onDrop = useCallback((e) => {
    e.preventDefault();
    dropZoneRef.current?.classList.remove('border-cyan-500');
    const file = e.dataTransfer.files[0];
    if (file) setUploadedVideo(file);
  }, []);
  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setUploadedVideo(file);
  };

  const handleUpload = async () => {
    if (!uploadedVideo) return;
    setUploading(true);
    setUploadError('');

    const formData = new FormData();
    formData.append('file', uploadedVideo);

    try {
      const res = await fetch(`${API}/analyze-video`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) {
        setUploading(false);
        setUploadError(data.error);
        return;
      }
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const url = URL.createObjectURL(uploadedVideo);
      setPreviewUrl(url);
      if (data.meta) setCurrentVideoMeta(data.meta);
      if (data.decision) setMarlDecision(data.decision);
      setUploading(false);
    } catch (err) {
      setUploading(false);
      setUploadError('Upload failed: ' + err.message);
    }
  };

  // ── Active GPU selection for drill-down ────────────────────────────────────
  const activeGpu = useMemo(() => {
    return gpus.find(g => g.id === selectedGpuId) || gpus[0] || {};
  }, [gpus, selectedGpuId]);

  // Filtered Events
  const filteredLogs = useMemo(() => {
    if (eventFilter === 'risk') {
      return logs.filter(l => l.type === 'warning' || l.message?.toLowerCase().includes('risk') || l.message?.toLowerCase().includes('threshold'));
    }
    if (eventFilter === 'error') {
      return logs.filter(l => l.type === 'error' || l.message?.toLowerCase().includes('crash') || l.message?.toLowerCase().includes('quarantine'));
    }
    if (eventFilter === 'decision') {
      return logs.filter(l => l.message?.toLowerCase().includes('marl') || l.message?.toLowerCase().includes('scheduled') || l.message?.toLowerCase().includes('assigned'));
    }
    return logs;
  }, [logs, eventFilter]);

  // Candidate scores for the latest MARL decision
  const candidateScoresData = useMemo(() => {
    if (!marlDecision?.candidates || !Array.isArray(marlDecision.candidates)) return [];
    return marlDecision.candidates.map(c => ({
      ...c,
      displayScore: c.eligible ? Math.max(0, c.bid_score) : 0,
      isSelected: marlDecision.selected_gpu === c.label || marlDecision.selected_gpu_id === c.gpu_id,
    }));
  }, [marlDecision]);

  return (
    <div className="min-h-screen bg-[#030712] text-slate-100 p-3 sm:p-5 font-sans space-y-4">
      {/* ── TOP HEADER ──────────────────────────────────────────────────────── */}
      <header className="glass-panel p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-cyan-600 to-emerald-400 p-[1px] flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-slate-950 rounded-2xl flex items-center justify-center">
              <span className="text-cyan-400 text-xl font-black">⬡</span>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-black tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-400 bg-clip-text text-transparent">
                AETHERGRID
              </h1>
            </div>
            <p className="text-xs text-slate-400 font-medium">
              Real-Time Risk-Aware MARL Scheduling & Distributed Hardware Telemetry
            </p>
          </div>
        </div>

        {/* View Switcher Tabs & Status */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="bg-slate-950/80 border border-white/10 rounded-xl p-1 flex gap-1 text-xs">
            <button
              onClick={() => setActiveTab('bit8')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${
                activeTab === 'bit8'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>🌟</span>
              <span>FULL DASHBOARD</span>
            </button>
            <button
              onClick={() => setActiveTab('monitor')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${
                activeTab === 'monitor'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>🛡️</span>
              <span>Risk & MARL Monitor</span>
            </button>
            <button
              onClick={() => setActiveTab('workload')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${
                activeTab === 'workload'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>🎬</span>
              <span>Video Hub & Allocator</span>
            </button>
            <button
              onClick={() => setActiveTab('unified')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${
                activeTab === 'unified'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>⚡</span>
              <span>Unified Command</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${
                activeTab === 'benchmarks'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/20'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>🔬</span>
              <span>Research Benchmarks</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 border ${
              systemStatus === 'Connected'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
            }`}>
              <span className={`w-2 h-2 rounded-full ${systemStatus === 'Connected' ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
              {systemStatus}
            </span>
            <span className="px-3 py-1.5 rounded-xl text-xs bg-slate-900 border border-white/10 text-slate-300 font-mono font-medium">
              Mode: <strong className="text-cyan-400">{scheduler.toUpperCase()}</strong>
            </span>
          </div>
        </div>
      </header>

      {/* ── BIT 8 MASTER DASHBOARD SECTION ─────────────────────────────────── */}
      {activeTab === 'bit8' && (
        <Bit8Dashboard
          gpus={gpus}
          metrics={metrics}
          events={logs}
          tasks={tasksList}
          videoQueue={videoQueue}
          marlDecision={marlDecision}
          decisionHistory={decisionHistory}
          riskSummary={riskSummary}
          riskHistory={riskHistory}
          selectedGpuId={selectedGpuId}
          onSelectGpu={setSelectedGpuId}
          lastUpdated={lastUpdatedTime}
          isDisconnected={systemStatus !== 'Connected'}
          isRunning={isRunning}
          onStartSimulation={() => post('/start')}
          onPauseSimulation={() => post('/pause')}
          onStepSimulation={() => post('/step')}
          onResetSimulation={() => post('/reset')}
          onSubmitTask={async (taskParams) => {
            try {
              const res = await fetch(`${API}/api/tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(taskParams),
              });
              const data = await res.json();
              if (res.ok) {
                fetchInitialSnapshots();
              }
              return data;
            } catch (err) {
              console.error('Task submission error:', err);
              throw err;
            }
          }}
          onRefresh={fetchInitialSnapshots}
          scheduler={scheduler}
          onSetScheduler={applyScheduler}
        />
      )}

      {/* ── BIT 9 RESEARCH BENCHMARKING SECTION ───────────────────────────────── */}
      {activeTab === 'benchmarks' && (
        <ResearchBenchmarking />
      )}

      {/* ── REAL-TIME RISK & SCHEDULING MONITOR SECTION ──────────────────────── */}
      {(activeTab === 'monitor' || activeTab === 'unified') && (
        <div className="space-y-4 animate-fadeIn">
          {/* Section Header */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 px-1">
            <div className="flex items-center gap-2">
              <span className="text-cyan-400 text-lg">🛡️</span>
              <h2 className="text-base sm:text-lg font-bold text-slate-100 tracking-tight">
                REAL-TIME RISK & SCHEDULING MONITOR
              </h2>
              <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-white/10 font-mono">
                LIVE HARDWARE TELEMETRY & DECISION INFERENCE
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-cyan-400 inline-block" /> Avg Risk
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-amber-400 inline-block" /> Max Risk
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-purple-400 inline-block" /> Expected Failure Cost
              </span>
            </div>
          </div>

          {/* 3.A SUMMARY METRIC CARDS */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricCard
              label="Cluster Avg Risk"
              value={(riskSummary.cluster_avg_risk ?? 0).toFixed(4)}
              sub={getRiskLabel(riskSummary.cluster_avg_risk ?? 0)}
              denominator="Across 4 active GPUs"
              accent={riskSummary.cluster_avg_risk >= 0.75 ? 'rose' : riskSummary.cluster_avg_risk >= 0.5 ? 'amber' : 'cyan'}
              icon="📉"
              alertBadge={riskSummary.cluster_avg_risk >= 0.75 ? 'HIGH RISK' : null}
            />
            <MetricCard
              label="Maximum GPU Risk"
              value={(riskSummary.max_gpu_risk ?? 0).toFixed(4)}
              sub={`GPU ${riskSummary.max_risk_gpu_id || 1} (${getRiskLabel(riskSummary.max_gpu_risk ?? 0)})`}
              denominator="Peak failure vulnerability"
              accent={riskSummary.max_gpu_risk >= 0.85 ? 'rose' : riskSummary.max_gpu_risk >= 0.75 ? 'amber' : 'cyan'}
              icon="🔥"
              alertBadge={riskSummary.max_gpu_risk >= 0.85 ? 'QUARANTINE' : riskSummary.max_gpu_risk >= 0.75 ? 'WARNING' : null}
            />
            <MetricCard
              label="Avg Expected Failure Cost"
              value={`${(riskSummary.avg_efc ?? 0).toFixed(4)}`}
              sub="Impact units / task"
              denominator="EFC = P(fail) × Task Severity"
              accent="purple"
              icon="⚖️"
            />
            <MetricCard
              label="Risk-Affected Decisions"
              value={riskSummary.risk_affected_decisions ?? 0}
              sub="Ranking/Selection Pivots"
              denominator="Live MARL bidding penalty"
              accent="amber"
              icon="🧠"
            />
            <MetricCard
              label="Quarantines / Cooldowns"
              value={`${riskSummary.quarantined_count ?? 0} / ${riskSummary.cooldown_count ?? 0}`}
              sub="Quarantined / Cooldown"
              denominator="Fail-safe isolation protection"
              accent={riskSummary.quarantined_count > 0 ? 'rose' : riskSummary.cooldown_count > 0 ? 'amber' : 'emerald'}
              icon="🛡️"
              alertBadge={riskSummary.quarantined_count > 0 ? 'ACTIVE ISOLATION' : null}
            />
            <MetricCard
              label="Scheduling Success Rate"
              value={`${(riskSummary.scheduling_success_rate ?? 100.0).toFixed(1)}%`}
              sub="Accepted & Completed"
              denominator="Denominator: Submitted Tasks"
              accent="emerald"
              icon="🎯"
            />
          </div>

          {/* 3.B LIVE RISK TIME-SERIES CHART & 3.D RISK-FACTOR BREAKDOWN */}
          <div className="grid grid-cols-12 gap-4">
            {/* Live Risk Time-Series Chart */}
            <div className="col-span-12 lg:col-span-8 glass-panel p-4 space-y-3 border border-white/10">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-cyan-400">📈</span>
                  <h3 className="text-sm font-bold text-slate-100">
                    Live Cluster Risk & Expected Failure Cost (EFC) Rolling Trajectory
                  </h3>
                </div>
                <div className="flex items-center gap-2 text-[10px] font-mono">
                  <span className="text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-white/10">
                    Window: Last {riskHistory.length || 0} Telemetry Samples
                  </span>
                  <span className="text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/30">
                    Warn: 0.75
                  </span>
                  <span className="text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/30">
                    Quarantine: 0.85
                  </span>
                </div>
              </div>

              {riskHistory.length > 0 ? (
                <div className="w-full h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={riskHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="avgRiskGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.0} />
                        </linearGradient>
                        <linearGradient id="maxRiskGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#fb923c" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#fb923c" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                      <XAxis dataKey="time_str" tick={{ fill: '#64748b', fontSize: 10 }} />
                      <YAxis domain={[0, 1.0]} tick={{ fill: '#64748b', fontSize: 10 }} />
                      <Tooltip content={<CustomChartTooltip />} />
                      <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />

                      {/* Threshold Reference Lines */}
                      <ReferenceLine y={0.75} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'Warning Threshold (0.75)', fill: '#f59e0b', fontSize: 10 }} />
                      <ReferenceLine y={0.85} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Quarantine Threshold (0.85)', fill: '#ef4444', fontSize: 10 }} />

                      <Area type="monotone" name="Cluster Avg Risk" dataKey="cluster_avg_risk" stroke="#22d3ee" strokeWidth={2} fillOpacity={1} fill="url(#avgRiskGrad)" />
                      <Area type="monotone" name="Max GPU Risk" dataKey="max_gpu_risk" stroke="#fb923c" strokeWidth={2} fillOpacity={1} fill="url(#maxRiskGrad)" />
                      <Line type="monotone" name="Avg EFC" dataKey="avg_efc" stroke="#a78bfa" strokeWidth={2} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-slate-500 text-xs gap-2">
                  <span className="text-2xl animate-spin">⏳</span>
                  <span>Awaiting initial telemetry stream... Click "▷ Start" below to initiate workload.</span>
                </div>
              )}
            </div>

            {/* 3.D Risk-Factor Breakdown (Radar & Bar breakdown for Active GPU) */}
            <div className="col-span-12 lg:col-span-4 glass-panel p-4 space-y-3 border border-white/10 flex flex-col justify-between">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className="text-cyan-400">🔬</span>
                  <h3 className="text-sm font-bold text-slate-100">
                    Risk-Factor Breakdown — GPU {activeGpu.id}
                  </h3>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                  (activeGpu.risk_score ?? 0) >= 0.75 ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                  (activeGpu.risk_score ?? 0) >= 0.40 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                  'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`}>
                  {getRiskLabel(activeGpu.risk_score ?? 0)}: {(activeGpu.risk_score ?? 0).toFixed(4)}
                </span>
              </div>

              {/* Radar Chart */}
              <RiskRadarChart components={activeGpu.risk_components} gpuLabel={`GPU ${activeGpu.id}`} />

              {/* Key Factor Bars */}
              {activeGpu.risk_components ? (
                <div className="grid grid-cols-2 gap-2 text-[10px] font-mono pt-1 border-t border-white/5">
                  <div className="bg-slate-950/60 p-1.5 rounded-lg border border-white/5 flex justify-between">
                    <span className="text-slate-400">Temp Risk:</span>
                    <span className="text-cyan-300 font-bold">{((activeGpu.risk_components.temperature_risk ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded-lg border border-white/5 flex justify-between">
                    <span className="text-slate-400">Mem Pressure:</span>
                    <span className="text-cyan-300 font-bold">{((activeGpu.risk_components.memory_pressure ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded-lg border border-white/5 flex justify-between">
                    <span className="text-slate-400">Frag Risk:</span>
                    <span className="text-cyan-300 font-bold">{((activeGpu.risk_components.fragmentation_risk ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded-lg border border-white/5 flex justify-between">
                    <span className="text-slate-400">Anomaly:</span>
                    <span className="text-cyan-300 font-bold">{((activeGpu.risk_components.anomaly_score ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ) : (
                <p className="text-[10px] text-slate-500 text-center">Select any GPU below to inspect risk factors.</p>
              )}
            </div>
          </div>

          {/* 3.C PER-GPU RISK COMPARISON CARDS */}
          <div className="space-y-2">
            <div className="flex justify-between items-center px-1">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <span>🖥️</span> Per-GPU Hardware Health & Eligibility Matrix (Click GPU to Drill Down)
              </h3>
              <span className="text-[10px] text-slate-500">Live 4-GPU Partition Telemetry</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {gpus.map((gpu) => {
                const health = getHealthStatus(gpu);
                const isSelected = selectedGpuId === gpu.id;
                const isAssigned = marlDecision?.selected_gpu === `GPU ${gpu.id}` || marlDecision?.selected_gpu_id === gpu.id;

                return (
                  <div
                    key={gpu.id}
                    onClick={() => setSelectedGpuId(gpu.id)}
                    className={`cursor-pointer bg-slate-900/70 border rounded-2xl p-3.5 space-y-2.5 transition-all duration-300 ${
                      isSelected
                        ? 'border-cyan-400 ring-2 ring-cyan-500/20 bg-slate-900 shadow-[0_0_25px_rgba(34,211,238,0.15)]'
                        : isAssigned
                        ? 'border-emerald-500/60 ring-1 ring-emerald-500/30'
                        : 'border-white/10 hover:border-white/20'
                    }`}
                  >
                    {/* Header */}
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span className="font-black text-slate-100 text-sm">GPU {gpu.id}</span>
                        {isAssigned && (
                          <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
                            ✓ LAST ALLOCATED
                          </span>
                        )}
                      </div>
                      <span className={`text-[9px] px-2 py-0.5 rounded-full border font-bold ${health.cls}`}>
                        {health.label}
                      </span>
                    </div>

                    {/* Risk & EFC Badges */}
                    <div className="grid grid-cols-2 gap-1.5 text-xs font-mono">
                      <div className="bg-slate-950/80 p-2 rounded-xl border border-white/5 flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase">Risk Score</span>
                        <span className={`font-bold text-sm ${getRiskColor(gpu.risk_score ?? 0) === '#ef4444' ? 'text-rose-400' : getRiskColor(gpu.risk_score ?? 0) === '#f97316' ? 'text-amber-400' : 'text-cyan-400'}`}>
                          {(gpu.risk_score ?? 0).toFixed(4)}
                        </span>
                        <span className="text-[8px] text-slate-500 uppercase">{getRiskLabel(gpu.risk_score ?? 0)}</span>
                      </div>
                      <div className="bg-slate-950/80 p-2 rounded-xl border border-white/5 flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase">EFC Cost</span>
                        <span className="font-bold text-sm text-purple-400">
                          {(gpu.efc ?? 0).toFixed(4)}
                        </span>
                        <span className="text-[8px] text-slate-500 uppercase">Per Task</span>
                      </div>
                    </div>

                    {/* VRAM Progress Bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                        <span>VRAM Allocated</span>
                        <span>{((8 * (gpu.memory ?? 0)) / 100).toFixed(1)} / 8.0 GB ({(gpu.memory ?? 0).toFixed(0)}%)</span>
                      </div>
                      <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            gpu.memory > 85 ? 'bg-rose-500' : gpu.memory > 65 ? 'bg-amber-500' : 'bg-cyan-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(0, gpu.memory ?? 0))}%` }}
                        />
                      </div>
                    </div>

                    {/* Telemetry Row */}
                    <div className="grid grid-cols-3 gap-1 text-[10px] font-mono text-center">
                      <div className="bg-slate-950/50 p-1 rounded-lg border border-white/5">
                        <span className="text-slate-500 block text-[9px]">TEMP</span>
                        <span className={`font-bold ${tempColor(gpu.temperature)}`}>{gpu.temperature}°C</span>
                      </div>
                      <div className="bg-slate-950/50 p-1 rounded-lg border border-white/5">
                        <span className="text-slate-500 block text-[9px]">FRAG</span>
                        <span className="font-bold text-slate-300">{(gpu.fragmentation ?? 0).toFixed(1)}%</span>
                      </div>
                      <div className="bg-slate-950/50 p-1 rounded-lg border border-white/5">
                        <span className="text-slate-500 block text-[9px]">QUEUE</span>
                        <span className="font-bold text-slate-300">{gpu.queue_length ?? 0}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 3.E MARL DECISION ANALYSIS & 3.F RISK TIMELINE */}
          <div className="grid grid-cols-12 gap-4">
            {/* MARL Decision Analysis Panel */}
            <div className="col-span-12 lg:col-span-7 glass-panel p-4 space-y-3 border border-emerald-500/20">
              <div className="flex justify-between items-center border-b border-white/10 pb-2">
                <div className="flex items-center gap-2">
                  <span className="text-emerald-400">🧠</span>
                  <h3 className="text-sm font-bold text-slate-100">
                    Latest MARL Decision Deep-Dive & Candidate Score Decomposition
                  </h3>
                </div>
                {marlDecision && (
                  <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                    {marlDecision.decision_id || `Task: ${marlDecision.task_id}`}
                  </span>
                )}
              </div>

              {marlDecision ? (
                <div className="space-y-3">
                  {/* Selected GPU Hero & Decision Context */}
                  <div className="bg-slate-950/90 border border-emerald-500/30 rounded-2xl p-3.5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Assigned GPU:</span>
                        <span className="text-xl font-black text-emerald-400">
                          {marlDecision.selected_gpu || (marlDecision.selected_gpu_id ? `GPU ${marlDecision.selected_gpu_id}` : 'None (Rejected)')}
                        </span>
                        <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                          marlDecision.status === 'assigned' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                        }`}>
                          {marlDecision.status?.toUpperCase() || 'EVALUATED'}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 font-medium">
                        Task: <strong className="text-cyan-400 font-mono">{marlDecision.task_id}</strong> · Mem: <strong className="text-slate-100 font-mono">{marlDecision.task_memory} GB</strong> · Time: <strong className="text-slate-100 font-mono">{marlDecision.task_time}s</strong>
                      </p>
                      {marlDecision.reason && (
                        <p className="text-[11px] text-slate-400 italic">
                          "{marlDecision.reason}"
                        </p>
                      )}
                    </div>

                    <div className="flex sm:flex-col gap-2 sm:gap-1 text-right text-xs font-mono">
                      <div>
                        <span className="text-slate-500 text-[10px]">Winning Bid Score: </span>
                        <span className="text-emerald-400 font-bold text-sm">
                          {typeof marlDecision.decision_score === 'number' ? marlDecision.decision_score.toFixed(4) : '—'}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500 text-[10px]">Assigned Risk / EFC: </span>
                        <span className="text-amber-400 font-bold">{(marlDecision.risk_score ?? 0).toFixed(3)}</span>
                        <span className="text-slate-500"> / </span>
                        <span className="text-purple-400 font-bold">{(marlDecision.expected_failure_cost ?? 0).toFixed(3)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Candidate Bidding Comparison Ranked Chart */}
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                      Candidate GPU Bid Scores & Penalty Decomposition
                    </span>
                    {candidateScoresData.length > 0 ? (
                      <div className="w-full h-36">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={candidateScoresData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
                            <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
                            <YAxis type="category" dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 'bold' }} />
                            <Tooltip content={<CustomDecisionTooltip />} />
                            <Bar dataKey="displayScore" name="Bid Score" radius={[0, 4, 4, 0]}>
                              {candidateScoresData.map((entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={entry.isSelected ? '#34d399' : entry.eligible ? '#22d3ee' : '#f43f5e'}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="text-center py-4 text-xs text-slate-500">
                        No candidate breakdown available.
                      </div>
                    )}
                  </div>

                  {/* Detailed Candidate Breakdown Table */}
                  {candidateScoresData.length > 0 && (
                    <div className="overflow-x-auto rounded-xl border border-white/5 bg-slate-950/70">
                      <table className="w-full text-left text-[11px] font-mono">
                        <thead className="bg-slate-900/90 text-slate-400 uppercase text-[9px] border-b border-white/10">
                          <tr>
                            <th className="p-2">GPU</th>
                            <th className="p-2">Status</th>
                            <th className="p-2">Perf Score</th>
                            <th className="p-2">Risk Penalty</th>
                            <th className="p-2">EFC Penalty</th>
                            <th className="p-2">Q-Val</th>
                            <th className="p-2 text-right">Final Bid</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {candidateScoresData.map((c) => (
                            <tr key={c.gpu_id} className={c.isSelected ? 'bg-emerald-500/10 text-emerald-300 font-bold' : 'text-slate-300'}>
                              <td className="p-2">{c.label} {c.isSelected ? '✓' : ''}</td>
                              <td className="p-2">
                                <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                                  c.eligible ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                                }`}>
                                  {c.eligible ? 'ELIGIBLE' : c.ineligible_reason || 'INELIGIBLE'}
                                </span>
                              </td>
                              <td className="p-2 text-cyan-400">{c.perf_score?.toFixed(4) ?? '0.0000'}</td>
                              <td className="p-2 text-rose-400">-{c.risk_penalty?.toFixed(4) ?? '0.0000'}</td>
                              <td className="p-2 text-amber-400">-{c.efc_penalty?.toFixed(4) ?? '0.0000'}</td>
                              <td className="p-2 text-purple-400">+{c.q_contribution?.toFixed(4) ?? '0.0000'}</td>
                              <td className="p-2 text-right font-bold text-white">
                                {c.bid_score > -50 ? c.bid_score.toFixed(4) : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Compact Recent Decisions History */}
                  <div className="space-y-1.5 pt-2 border-t border-white/5">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                      Recent Decisions History (Chronological)
                    </span>
                    <div className="max-h-28 overflow-y-auto rounded-xl border border-white/5 bg-slate-950/60 divide-y divide-white/5 text-[10px] font-mono">
                      {decisionHistory.slice(0, 8).map((dec, i) => (
                        <div key={dec.decision_id || i} className="p-1.5 px-2 flex justify-between items-center text-slate-300">
                          <span className="text-cyan-400">{dec.task_id}</span>
                          <span>{dec.selected_gpu || 'Rejected'}</span>
                          <span className="text-emerald-400">Score: {typeof dec.decision_score === 'number' ? dec.decision_score.toFixed(3) : '—'}</span>
                          <span className="text-amber-400">Risk: {(dec.risk_score ?? 0).toFixed(3)}</span>
                          <span className={dec.status === 'assigned' ? 'text-emerald-400' : 'text-rose-400'}>{dec.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-xs gap-1">
                  <span className="text-2xl">🎯</span>
                  <span>Awaiting initial scheduling decision. Submit a video or start simulation.</span>
                </div>
              )}
            </div>

            {/* 3.F Risk Events & Execution Timeline */}
            <div className="col-span-12 lg:col-span-5 glass-panel p-4 space-y-3 border border-white/10 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-center border-b border-white/10 pb-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-amber-400">⚡</span>
                    <h3 className="text-sm font-bold text-slate-100">
                      Risk Events & Execution Timeline
                    </h3>
                  </div>
                  <span className="text-[10px] bg-slate-950 px-2 py-0.5 rounded text-slate-400 border border-white/10 font-mono">
                    {filteredLogs.length} events
                  </span>
                </div>

                {/* Filter Tabs */}
                <div className="flex gap-1 mb-2 text-[10px] font-mono">
                  {[
                    ['all', 'All'],
                    ['risk', 'Risk & Warn'],
                    ['error', 'Faults'],
                    ['decision', 'Scheduler'],
                  ].map(([key, lbl]) => (
                    <button
                      key={key}
                      onClick={() => setEventFilter(key)}
                      className={`px-2 py-1 rounded-md transition-all ${
                        eventFilter === key ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold' : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>

                {/* Timeline List */}
                <div className="h-80 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
                  {filteredLogs.map((log, idx) => {
                    const isError = log.type === 'error' || log.message?.includes('threshold QUARANTINE') || log.message?.includes('CRASH');
                    const isWarn = log.type === 'warning' || log.message?.includes('threshold WARNING') || log.message?.includes('cooldown');
                    const isSuccess = log.type === 'success' || log.message?.includes('completed') || log.message?.includes('recovered');

                    return (
                      <div
                        key={log.id || idx}
                        className={`p-2 rounded-xl border flex flex-col gap-0.5 transition-all ${
                          isError
                            ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                            : isWarn
                            ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                            : isSuccess
                            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                            : 'bg-slate-950/70 border-white/5 text-slate-300'
                        }`}
                      >
                        <div className="flex justify-between items-center text-[9px] text-slate-500">
                          <span>[{log.time || log.timestamp?.split('T')[1]?.slice(0, 8) || '00:00:00'}]</span>
                          <span className="uppercase tracking-wider">{log.type || 'INFO'}</span>
                        </div>
                        <p className="font-sans text-xs font-medium leading-relaxed">
                          {log.message}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="pt-2 border-t border-white/5 text-[10px] text-slate-500 flex justify-between">
                <span>Deduplicated event stream</span>
                <span>Auto-reconnecting WebSocket</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── VIDEO WORKLOAD HUB & ALLOCATOR SECTION ──────────────────────────── */}
      {(activeTab === 'workload' || activeTab === 'unified') && (
        <div className="grid grid-cols-12 gap-4 animate-fadeIn">
          {/* Controls & Queue Sidebar */}
          <aside className="col-span-12 lg:col-span-4 space-y-4">
            {/* Simulation Controls */}
            <div className="glass-panel p-4 space-y-3 border border-white/10">
              <h3 className="text-sm font-bold text-slate-100 border-b border-white/10 pb-2 flex items-center gap-2">
                <span>⚙️</span> Simulation Controls & Schedulers
              </h3>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <label className="text-[11px] text-slate-400">Scheduler Algorithm</label>
                  <select
                    value={scheduler}
                    onChange={(e) => {
                      setScheduler(e.target.value);
                      applyScheduler(e.target.value);
                    }}
                    className="w-full bg-slate-950 border border-white/10 rounded-xl p-2 text-xs text-cyan-300 font-bold outline-none focus:border-cyan-500"
                  >
                    <option value="marl">MARL (Risk-Aware)</option>
                    <option value="rl">Single RL Agent</option>
                    <option value="fcfs">FCFS</option>
                    <option value="round_robin">Round Robin</option>
                    <option value="least_loaded">Least Loaded</option>
                    <option value="random">Random</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-slate-400">Simulation Speed</label>
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(e.target.value)}
                    className="w-full bg-slate-950 border border-white/10 rounded-xl p-2 text-xs text-slate-200 outline-none focus:border-cyan-500"
                  >
                    <option>Slow</option>
                    <option>Medium</option>
                    <option>Fast</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <button
                  onClick={() => {
                    applyScheduler();
                    post('/start');
                  }}
                  className="bg-emerald-950/70 border border-emerald-500/40 hover:bg-emerald-900/60 text-emerald-300 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all shadow-lg shadow-emerald-900/20"
                >
                  <span>▷</span> Start Simulation
                </button>
                <button
                  onClick={() => post('/pause')}
                  className="bg-amber-950/70 border border-amber-500/40 hover:bg-amber-900/60 text-amber-300 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all"
                >
                  <span>⏸</span> Pause
                </button>
              </div>
              <button
                onClick={() => post('/reset')}
                className="w-full bg-slate-950 border border-white/10 hover:bg-slate-900 text-slate-300 py-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
              >
                <span>⟳</span> Reset Cluster & Queues
              </button>
            </div>

            {/* Video Queue */}
            <div className="glass-panel p-4 space-y-2 border border-white/10">
              <div className="flex justify-between items-center border-b border-white/10 pb-2">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <span>📂</span> Video Task Arrival Queue
                </h3>
                <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/30">
                  {videoQueue.length} queued
                </span>
              </div>

              {videoQueue.length === 0 ? (
                <p className="text-xs text-slate-600 text-center py-4">No tasks waiting in queue.</p>
              ) : (
                <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                  {videoQueue.map((t) => (
                    <div key={t.id} className="bg-slate-950/80 rounded-xl p-2.5 border border-white/5 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-cyan-400 text-xs font-mono font-bold">{t.id}</span>
                        <span className={`text-[9px] px-2 py-0.5 rounded-full border font-bold ${severityColor[t.workload_severity] || 'text-slate-400'}`}>
                          {t.workload_severity}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        {t.resolution} · {t.fps} FPS · {t.duration}s
                      </div>
                      <div className="text-[11px] text-amber-400 font-mono font-medium">
                        {t.memory_required} GB VRAM required
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>

          {/* Video Input Dropzone & Live Stream Preview */}
          <div className="col-span-12 lg:col-span-8 space-y-4">
            {/* Upload Section */}
            <section className="glass-panel p-4 space-y-3 border border-cyan-500/30">
              <div className="flex items-center gap-2">
                <span className="text-cyan-400 text-lg">🎬</span>
                <h3 className="text-sm font-bold text-slate-100">DYNAMIC VIDEO WORKLOAD INGESTION</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
                <div
                  ref={dropZoneRef}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className="md:col-span-6 border-2 border-dashed border-white/10 rounded-2xl p-5 flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-cyan-500/50 hover:bg-slate-950/40 transition-all text-center"
                >
                  <span className="text-3xl">📁</span>
                  <p className="text-xs text-slate-300 font-medium">
                    Drag & drop video or <span className="text-cyan-400 underline font-bold">browse</span>
                  </p>
                  <p className="text-[10px] text-slate-500 font-mono">MP4 · AVI · MOV · MKV · WEBM</p>
                  <input ref={fileInputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm" onChange={onFileChange} className="hidden" />
                </div>

                <div className="md:col-span-6 flex flex-col justify-between gap-2">
                  {uploadedVideo ? (
                    <div className="bg-slate-950/90 rounded-2xl p-3 border border-white/10 space-y-1.5 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">File:</span>
                        <span className="text-cyan-400 font-bold truncate max-w-[180px]">{uploadedVideo.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Size:</span>
                        <span className="text-slate-200">{(uploadedVideo.size / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Status:</span>
                        <span className={uploading ? 'text-amber-400 font-bold animate-pulse' : 'text-emerald-400 font-bold'}>
                          {uploading ? '⏳ Analyzing & Ingesting...' : '✅ Ready to Schedule'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-950/40 rounded-2xl p-4 border border-white/5 text-xs text-slate-600 flex items-center justify-center h-24">
                      No video selected. Upload a clip to trigger live MARL bidding.
                    </div>
                  )}

                  {uploadError && (
                    <p className="text-[11px] text-rose-400 bg-rose-500/10 border border-rose-500/30 rounded-xl p-2">
                      {uploadError}
                    </p>
                  )}

                  <button
                    onClick={handleUpload}
                    disabled={!uploadedVideo || uploading}
                    className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold py-2.5 rounded-xl text-xs transition-all shadow-lg shadow-cyan-600/20 flex items-center justify-center gap-2"
                  >
                    {uploading ? '⏳ Ingesting Video Workload...' : '🚀 Ingest & Schedule via Risk-Aware MARL'}
                  </button>
                </div>
              </div>
            </section>

            {/* Live Stream Preview & Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <section className="glass-panel p-4 space-y-2 border border-white/10">
                <div className="flex items-center gap-2">
                  <span className="text-red-500 animate-pulse font-black">●</span>
                  <h3 className="text-sm font-bold text-slate-100">Live Video Stream Preview</h3>
                </div>

                {previewUrl ? (
                  <video
                    key={previewUrl}
                    src={previewUrl}
                    controls
                    muted
                    autoPlay
                    loop
                    className="w-full rounded-xl border border-white/10 max-h-44 object-cover"
                  />
                ) : (
                  <div className="w-full h-36 rounded-xl border border-white/5 bg-slate-950/80 flex items-center justify-center text-slate-600 text-xs">
                    No video streaming currently.
                  </div>
                )}
              </section>

              <section className="glass-panel p-4 space-y-2 border border-white/10">
                <h3 className="text-sm font-bold text-slate-100">Video Metadata Extraction</h3>
                {currentVideoMeta ? (
                  <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono">
                    {[
                      ['File', currentVideoMeta.filename],
                      ['Resolution', `${currentVideoMeta.width} × ${currentVideoMeta.height}`],
                      ['FPS', currentVideoMeta.fps],
                      ['Duration', `${currentVideoMeta.duration} sec`],
                      ['Frames', currentVideoMeta.frames],
                      ['Codec', currentVideoMeta.codec],
                      ['Bitrate', currentVideoMeta.bitrate],
                      ['Memory Req', `${currentVideoMeta.memory_required ?? 2.0} GB`],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-slate-950/70 p-2 rounded-xl border border-white/5 flex flex-col">
                        <span className="text-slate-500 text-[9px] uppercase tracking-wider">{k}</span>
                        <span className="text-slate-200 font-bold truncate">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-600 text-center py-8">
                    Upload video above to inspect extracted resolution, FPS, and codec properties.
                  </p>
                )}
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
