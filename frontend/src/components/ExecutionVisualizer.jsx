import React, { useState, useMemo } from 'react';

const getEventBadge = (type, message = '') => {
  const t = String(type || 'INFO').toLowerCase();
  const m = message.toLowerCase();
  if (t === 'error' || m.includes('crash') || m.includes('failed')) {
    return { label: 'CRITICAL', cls: 'text-rose-400 bg-rose-500/10 border-rose-500/30' };
  }
  if (t === 'warning' || m.includes('preempt') || m.includes('warning') || m.includes('cooldown')) {
    return { label: 'WARNING', cls: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
  }
  if (t === 'success' || m.includes('completed') || m.includes('recovered')) {
    return { label: 'SUCCESS', cls: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
  }
  return { label: 'INFO', cls: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' };
};

export default function ExecutionVisualizer({
  gpus = [],
  events = [],
  metrics = {},
  tasks = [],
}) {
  const [eventCategory, setEventCategory] = useState('ALL');
  const [searchFilter, setSearchFilter] = useState('');

  // Execution overview counts from real backend metrics
  const execMetrics = useMemo(() => {
    const completed = metrics.completed ?? 0;
    const running = metrics.total_running_tasks ?? 0;
    const queued = metrics.total_queued_tasks ?? 0;
    const crashes = metrics.crashes ?? 0;
    const preemptions = metrics.total_preemptions ?? 0;
    const throughput = completed > 0 ? (completed / Math.max(1, (metrics.step ?? 1))).toFixed(2) : '0.00';

    return {
      completed,
      running,
      queued,
      crashes,
      preemptions,
      throughput,
      successRate: completed + crashes > 0 ? ((completed / (completed + crashes)) * 100).toFixed(1) : '100.0',
    };
  }, [metrics]);

  // Filtered execution events
  const filteredEvents = useMemo(() => {
    let list = events || [];
    if (eventCategory === 'FAULTS') {
      list = list.filter((e) => e.type === 'error' || e.message?.toLowerCase().includes('crash') || e.message?.toLowerCase().includes('fail'));
    } else if (eventCategory === 'WARN') {
      list = list.filter((e) => e.type === 'warning' || e.message?.toLowerCase().includes('preempt') || e.message?.toLowerCase().includes('threshold'));
    } else if (eventCategory === 'TASKS') {
      list = list.filter((e) => e.message?.toLowerCase().includes('task') || e.message?.toLowerCase().includes('completed') || e.message?.toLowerCase().includes('scheduled'));
    }

    if (searchFilter.trim()) {
      const q = searchFilter.toLowerCase();
      list = list.filter((e) => e.message?.toLowerCase().includes(q) || e.id?.toLowerCase().includes(q));
    }

    return list.slice(0, 60);
  }, [events, eventCategory, searchFilter]);

  // Terminal completed tasks from task list
  const recentCompletedTasks = useMemo(() => {
    if (!Array.isArray(tasks)) return [];
    return tasks
      .filter((t) => (t.status || '').toUpperCase() === 'COMPLETED')
      .slice(0, 15);
  }, [tasks]);

  return (
    <section className="glass-panel p-4 space-y-4 border border-white/10">
      {/* Title & Execution Overview */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-lg">⚡</span>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Real-Time Task Execution Monitoring & Timeline
            </h3>
            <p className="text-[11px] text-slate-400">
              Live task execution states, GPU core mappings, and timeline events
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="bg-slate-900 border border-white/10 text-cyan-400 px-2.5 py-1 rounded-xl">
            Success Rate: <strong className="text-emerald-400">{execMetrics.successRate}%</strong>
          </span>
        </div>
      </div>

      {/* 1. EXECUTION OVERVIEW CARDS */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 text-xs font-mono">
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">RUNNING TASKS</span>
          <p className="text-xl font-black text-cyan-400">{execMetrics.running}</p>
          <span className="text-[9px] text-slate-400">Executing on GPUs</span>
        </div>
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">QUEUED TASKS</span>
          <p className="text-xl font-black text-blue-400">{execMetrics.queued}</p>
          <span className="text-[9px] text-slate-400">Pending allocation</span>
        </div>
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">COMPLETED TASKS</span>
          <p className="text-xl font-black text-emerald-400">{execMetrics.completed}</p>
          <span className="text-[9px] text-slate-400">Successfully rendered</span>
        </div>
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">CRASH EVENTS</span>
          <p className="text-xl font-black text-rose-400">{execMetrics.crashes}</p>
          <span className="text-[9px] text-slate-400">GPU fault triggers</span>
        </div>
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">PREEMPTIONS</span>
          <p className="text-xl font-black text-amber-400">{execMetrics.preemptions}</p>
          <span className="text-[9px] text-slate-400">Dynamic evacuations</span>
        </div>
        <div className="bg-slate-950/80 p-3 rounded-xl border border-white/5 space-y-1">
          <span className="text-[10px] text-slate-500 uppercase">THROUGHPUT</span>
          <p className="text-xl font-black text-purple-400">{execMetrics.throughput}</p>
          <span className="text-[9px] text-slate-400">Tasks / sim step</span>
        </div>
      </div>

      {/* 2. ACTIVE GPU EXECUTION MATRIX & EVENT STREAM */}
      <div className="grid grid-cols-12 gap-4">
        {/* Active GPU Partition Running Status */}
        <div className="col-span-12 lg:col-span-5 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-3">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase">GPU Partition Active Execution</span>
            <span className="text-slate-500 text-[10px]">4 Dedicated Partitions</span>
          </div>

          <div className="space-y-2 font-mono text-xs">
            {gpus.map((gpu) => {
              const isRunning = (gpu.running_tasks ?? 0) > 0;
              const isCooldown = (gpu.cooldown ?? 0) > 0;
              const isCrashed = Boolean(gpu.crashed);
              const isQuarantined = Boolean(gpu.is_quarantined);

              return (
                <div
                  key={gpu.id}
                  className="bg-slate-900/60 p-2.5 rounded-xl border border-white/5 flex flex-col gap-1.5"
                >
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-slate-200">GPU {gpu.id}</span>
                    <span
                      className={`text-[9px] px-2 py-0.5 rounded-full border font-bold ${
                        isCrashed
                          ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                          : isQuarantined
                          ? 'bg-purple-500/20 text-purple-400 border-purple-500/30'
                          : isCooldown
                          ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                          : isRunning
                          ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 animate-pulse'
                          : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      }`}
                    >
                      {isCrashed
                        ? 'CRASHED'
                        : isQuarantined
                        ? 'QUARANTINED'
                        : isCooldown
                        ? `COOLDOWN (${gpu.cooldown}s)`
                        : isRunning
                        ? `EXECUTING (${gpu.running_tasks} tasks)`
                        : 'IDLE (READY)'}
                    </span>
                  </div>

                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>
                      VRAM: <strong className="text-white">{((8 * (gpu.memory ?? 0)) / 100).toFixed(1)} GB</strong> ({(gpu.memory ?? 0).toFixed(0)}%)
                    </span>
                    <span>
                      Load: <strong className="text-white">{gpu.utilization ?? 0}%</strong>
                    </span>
                    <span>
                      Queue: <strong className="text-cyan-400">{gpu.queue_length ?? 0}</strong>
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Completed Task Log */}
          {recentCompletedTasks.length > 0 && (
            <div className="pt-2 border-t border-white/5 space-y-1 text-[10px] font-mono">
              <span className="text-slate-400 uppercase font-bold block">
                Recently Completed Workloads
              </span>
              <div className="max-h-24 overflow-y-auto space-y-1 text-slate-400">
                {recentCompletedTasks.map((t) => (
                  <div key={t.task_id} className="flex justify-between items-center p-1 bg-slate-900/40 rounded">
                    <span className="text-emerald-400 font-bold">{t.task_id}</span>
                    <span>GPU {t.assigned_gpu ?? '—'}</span>
                    <span className="text-slate-300">{t.duration ? `${t.duration.toFixed(2)}s` : 'Completed'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Live Execution Event Feed */}
        <div className="col-span-12 lg:col-span-7 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-2 flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-2">
              <span className="font-bold text-slate-300 uppercase text-xs font-mono">
                Live Execution Event Timeline Stream
              </span>
              <div className="flex items-center gap-1 text-[10px] font-mono">
                {['ALL', 'TASKS', 'WARN', 'FAULTS'].map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setEventCategory(cat)}
                    className={`px-2 py-0.5 rounded border transition-all ${
                      eventCategory === cat
                        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold'
                        : 'bg-slate-900 text-slate-500 border-white/5 hover:text-slate-300'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            <div className="h-72 overflow-y-auto space-y-1.5 pr-1 font-mono text-[11px]">
              {filteredEvents.length > 0 ? (
                filteredEvents.map((evt, i) => {
                  const badge = getEventBadge(evt.type, evt.message);
                  return (
                    <div
                      key={evt.id || i}
                      className="p-2 rounded-xl bg-slate-900/60 border border-white/5 flex flex-col gap-0.5"
                    >
                      <div className="flex justify-between items-center text-[9px] text-slate-500">
                        <span>[{evt.time || evt.timestamp?.split('T')[1]?.slice(0, 8) || '00:00:00'}]</span>
                        <span className={`px-1.5 py-0.2 rounded border uppercase font-bold ${badge.cls}`}>
                          {badge.label}
                        </span>
                      </div>
                      <p className="font-sans text-xs text-slate-200">
                        {evt.message}
                      </p>
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-16 text-slate-600 text-xs">
                  No execution events recorded matching filter.
                </div>
              )}
            </div>
          </div>

          <div className="pt-2 border-t border-white/5 text-[10px] text-slate-500 font-mono flex justify-between">
            <span>Deduplicated execution events</span>
            <span>Buffered real-time log</span>
          </div>
        </div>
      </div>
    </section>
  );
}
