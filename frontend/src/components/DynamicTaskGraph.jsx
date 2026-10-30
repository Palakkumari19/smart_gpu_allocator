import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const STAGES = [
  { id: 'SUBMITTED', label: 'Submitted', color: '#64748b', icon: '📥' },
  { id: 'QUEUED', label: 'Queued', color: '#38bdf8', icon: '⏳' },
  { id: 'SCHEDULED', label: 'Scheduled', color: '#a78bfa', icon: '🧭' },
  { id: 'RUNNING', label: 'Running', color: '#34d399', icon: '⚡' },
  { id: 'COMPLETED', label: 'Completed', color: '#10b981', icon: '✅' },
  { id: 'FAILED', label: 'Failed / Rejected', color: '#f43f5e', icon: '❌' },
];

const getStatusBadge = (status) => {
  const s = String(status || 'QUEUED').toUpperCase();
  if (s === 'COMPLETED') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
  if (s === 'RUNNING') return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 animate-pulse';
  if (s === 'SCHEDULED') return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
  if (s === 'QUEUED') return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  if (s === 'FAILED' || s.includes('REJECT')) return 'bg-rose-500/20 text-rose-400 border-rose-500/30';
  if (s === 'CANCELLED') return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  return 'bg-slate-800 text-slate-300 border-white/10';
};

export default function DynamicTaskGraph({
  tasks = [],
  videoQueue = [],
  onSelectTask,
}) {
  const [filterStage, setFilterStage] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  // Deduplicate tasks and normalize states across session
  const processedTasks = useMemo(() => {
    const taskMap = new Map();

    // 1. Add active video queue tasks
    if (Array.isArray(videoQueue)) {
      videoQueue.forEach((t) => {
        if (!t?.id) return;
        taskMap.set(t.id, {
          id: t.id,
          status: 'QUEUED',
          gpu_id: t.gpu_id ?? null,
          memory_required: t.memory_required ?? 2.0,
          workload_severity: t.workload_severity ?? 'MEDIUM',
          resolution: t.resolution ?? '1080p',
          fps: t.fps ?? 30,
          duration: t.duration ?? 5.0,
          timestamp: t.created_time || t.timestamp || new Date().toISOString(),
        });
      });
    }

    // 2. Add / merge tasks from backend task registry / API
    if (Array.isArray(tasks)) {
      tasks.forEach((t) => {
        const id = t.task_id || t.id;
        if (!id) return;
        const existing = taskMap.get(id) || {};
        let rawStatus = (t.status || existing.status || 'QUEUED').toUpperCase();
        if (rawStatus === 'ASSIGNED') rawStatus = 'SCHEDULED';
        taskMap.set(id, {
          ...existing,
          id,
          status: rawStatus,
          gpu_id: t.assigned_gpu ?? t.gpu_id ?? existing.gpu_id ?? null,
          memory_required: t.requested_memory_gb ?? t.memory_required ?? existing.memory_required ?? 2.0,
          workload_severity: t.priority ?? t.workload_severity ?? existing.workload_severity ?? 'MEDIUM',
          submission_time: t.submission_time || existing.submission_time,
          start_time: t.start_time || existing.start_time,
          completion_time: t.completion_time || existing.completion_time,
          duration: t.duration ?? existing.duration ?? null,
          failure_reason: t.failure_reason || existing.failure_reason,
          timestamp: t.completion_time || t.start_time || t.submission_time || existing.timestamp,
        });
      });
    }

    return Array.from(taskMap.values());
  }, [tasks, videoQueue]);

  // Stage counts for pipeline indicators
  const stageCounts = useMemo(() => {
    const counts = {
      SUBMITTED: processedTasks.length,
      QUEUED: 0,
      SCHEDULED: 0,
      RUNNING: 0,
      COMPLETED: 0,
      FAILED: 0,
    };
    processedTasks.forEach((t) => {
      const s = t.status;
      if (s === 'QUEUED' || s === 'ARRIVING' || s === 'WAITING') counts.QUEUED++;
      else if (s === 'SCHEDULED' || s === 'ASSIGNED') counts.SCHEDULED++;
      else if (s === 'RUNNING') counts.RUNNING++;
      else if (s === 'COMPLETED') counts.COMPLETED++;
      else if (s === 'FAILED' || s === 'CANCELLED' || s.includes('REJECT')) counts.FAILED++;
      else counts.QUEUED++;
    });
    return counts;
  }, [processedTasks]);

  // Stage Distribution chart data
  const chartData = useMemo(() => {
    return [
      { name: 'Queued', count: stageCounts.QUEUED, color: '#38bdf8' },
      { name: 'Scheduled', count: stageCounts.SCHEDULED, color: '#a78bfa' },
      { name: 'Running', count: stageCounts.RUNNING, color: '#34d399' },
      { name: 'Completed', count: stageCounts.COMPLETED, color: '#10b981' },
      { name: 'Failed', count: stageCounts.FAILED, color: '#f43f5e' },
    ];
  }, [stageCounts]);

  // Filtered task items
  const filteredTasks = useMemo(() => {
    let list = processedTasks;
    if (filterStage !== 'ALL') {
      list = list.filter((t) => {
        if (filterStage === 'QUEUED') return ['QUEUED', 'ARRIVING', 'WAITING'].includes(t.status);
        if (filterStage === 'SCHEDULED') return ['SCHEDULED', 'ASSIGNED'].includes(t.status);
        if (filterStage === 'FAILED') return ['FAILED', 'CANCELLED'].includes(t.status) || t.status.includes('REJECT');
        return t.status === filterStage;
      });
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((t) => t.id.toLowerCase().includes(q) || String(t.gpu_id).includes(q));
    }
    return list.slice(0, 50); // limit to recent 50 for clean responsive rendering
  }, [processedTasks, filterStage, searchQuery]);

  return (
    <section className="glass-panel p-4 space-y-4 border border-white/10">
      {/* Title & Pipeline Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-lg">🔄</span>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Dynamic Task Lifecycle Graph & State Progression
            </h3>
            <p className="text-[11px] text-slate-400">
              Real-time progression tracking across asynchronous execution stages
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-slate-400 bg-slate-950 px-2.5 py-1 rounded-xl border border-white/10">
            Total Ingested: <strong className="text-cyan-400">{processedTasks.length}</strong>
          </span>
        </div>
      </div>

      {/* 1. VISUAL STATE TRANSITION PIPELINE */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        {STAGES.map((st, idx) => {
          const count = stageCounts[st.id] ?? 0;
          const isSelected = filterStage === st.id;
          return (
            <button
              key={st.id}
              onClick={() => setFilterStage(isSelected ? 'ALL' : st.id)}
              className={`p-3 rounded-xl border text-left transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-slate-800 border-cyan-400 ring-1 ring-cyan-400/30'
                  : 'bg-slate-950/70 border-white/5 hover:border-white/20'
              }`}
            >
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400 font-mono text-[10px] uppercase">{st.label}</span>
                <span>{st.icon}</span>
              </div>
              <div className="my-1">
                <span className="text-xl font-black font-mono" style={{ color: st.color }}>
                  {count}
                </span>
              </div>
              <div className="flex items-center gap-1 text-[9px] text-slate-500 font-mono">
                <span>Stage {idx + 1}</span>
                {idx < 4 && <span>➔</span>}
              </div>
            </button>
          );
        })}
      </div>

      {/* 2. STAGE DISTRIBUTION CHART & PIPELINE FLOW */}
      <div className="grid grid-cols-12 gap-4">
        {/* Stage distribution bar chart */}
        <div className="col-span-12 lg:col-span-4 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 flex flex-col justify-between">
          <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block mb-1">
            Task Volume by Lifecycle Stage
          </span>
          <div className="w-full h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} allowDecimals={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const item = payload[0]?.payload;
                    return (
                      <div className="bg-slate-900 border border-cyan-500/30 p-2 rounded-lg text-xs font-mono shadow-xl">
                        <span style={{ color: item.color }} className="font-bold">
                          {item.name}:{' '}
                        </span>
                        <span className="text-white">{item.count} tasks</span>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[10px] text-slate-500 font-mono text-center pt-1 border-t border-white/5">
            Real-time pipeline distribution
          </div>
        </div>

        {/* Live Task Nodes & Timeline Feed */}
        <div className="col-span-12 lg:col-span-8 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-2.5">
          {/* Filter & Search Bar */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
            <div className="flex items-center gap-1.5 flex-wrap text-[10px] font-mono">
              <span className="text-slate-400">Filter:</span>
              {['ALL', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED'].map((st) => (
                <button
                  key={st}
                  onClick={() => setFilterStage(st)}
                  className={`px-2 py-0.5 rounded-md border transition-all ${
                    filterStage === st
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold'
                      : 'bg-slate-900 text-slate-400 border-white/5 hover:text-slate-200'
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>
            <input
              type="text"
              placeholder="Search Task ID (e.g. video_001)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded-lg px-2.5 py-1 text-xs text-slate-200 placeholder-slate-600 outline-none focus:border-cyan-500 font-mono w-full sm:w-48"
            />
          </div>

          {/* Task Transition Nodes List */}
          <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 font-mono text-xs">
            {filteredTasks.length > 0 ? (
              filteredTasks.map((t) => {
                const isSelected = filterStage === t.status;
                const statusBadge = getStatusBadge(t.status);

                return (
                  <div
                    key={t.id}
                    onClick={() => onSelectTask && onSelectTask(t)}
                    className="p-2.5 rounded-xl border border-white/5 bg-slate-900/60 hover:bg-slate-900 hover:border-cyan-500/30 transition-all flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2"
                  >
                    {/* Task identity & requirements */}
                    <div className="flex items-center gap-2.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-100 font-mono text-xs">{t.id}</span>
                          <span className={`text-[9px] px-1.5 py-0.2 rounded border font-bold ${statusBadge}`}>
                            {t.status}
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-400">
                          {t.workload_severity} · {t.memory_required} GB VRAM · {t.resolution || '1080p'}
                        </span>
                      </div>
                    </div>

                    {/* GPU Assignment & Duration */}
                    <div className="flex items-center gap-3 text-right">
                      {t.gpu_id !== null && t.gpu_id !== undefined ? (
                        <span className="text-[10px] px-2 py-0.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                          Assigned: {String(t.gpu_id).toUpperCase().startsWith('GPU') ? t.gpu_id : `GPU ${t.gpu_id}`}
                        </span>
                      ) : (
                        <span className="text-[10px] px-2 py-0.5 rounded-lg bg-slate-800 text-slate-500 border border-white/5">
                          Unassigned (In Queue)
                        </span>
                      )}
                      {t.duration !== null && (
                        <span className="text-[10px] text-amber-400 font-bold">
                          {typeof t.duration === 'number' ? `${t.duration.toFixed(2)}s` : t.duration}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-xs text-slate-500 space-y-1">
                <span className="text-base block">⏳</span>
                <p className="font-semibold text-slate-400">
                  {processedTasks.length === 0
                    ? 'No tasks ingested yet.'
                    : 'No tasks matching the selected filter or search query.'}
                </p>
                <p className="text-[11px] text-slate-500">
                  {processedTasks.length === 0
                    ? 'Use the Simulation Control bar above to submit a real task or start the simulation workload.'
                    : 'Try clearing the search query or switching to the "ALL" filter tab.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
