import React, { useState } from 'react';
import GPUDashboard from './GPUDashboard';
import DynamicTaskGraph from './DynamicTaskGraph';
import RiskVisualization from './RiskVisualization';
import MARLDecisionVisualizer from './MARLDecisionVisualizer';
import ExecutionVisualizer from './ExecutionVisualizer';

const WORKLOAD_PROFILES = {
  Light: {
    label: 'Light (720p)',
    memory: 1.0,
    time: 3,
    priority: 'low',
    desc: 'Light 720p Video Transcode',
    color: '#34d399',
  },
  Medium: {
    label: 'Medium (1080p)',
    memory: 2.2,
    time: 5,
    priority: 'medium',
    desc: 'Medium 1080p Video Transcode',
    color: '#38bdf8',
  },
  Heavy: {
    label: 'Heavy (1440p)',
    memory: 3.8,
    time: 8,
    priority: 'high',
    desc: 'Heavy 1440p Video Transcode',
    color: '#fb923c',
  },
  Extreme: {
    label: 'Extreme (4K)',
    memory: 7.5,
    time: 12,
    priority: 'high',
    desc: 'Extreme 4K UHD Video Render',
    color: '#f43f5e',
  },
};

export default function Bit8Dashboard({
  gpus = [],
  metrics = {},
  events = [],
  tasks = [],
  videoQueue = [],
  marlDecision = null,
  decisionHistory = [],
  riskSummary = {},
  riskHistory = [],
  selectedGpuId = 1,
  onSelectGpu,
  lastUpdated,
  isDisconnected = false,
  isRunning = false,
  onStartSimulation,
  onPauseSimulation,
  onStepSimulation,
  onResetSimulation,
  onSubmitTask,
  onRefresh,
  scheduler = 'marl',
  onSetScheduler,
}) {
  const [activeSection, setActiveSection] = useState('ALL');
  const [selectedProfile, setSelectedProfile] = useState('Medium');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionFeedback, setSubmissionFeedback] = useState(null);

  const handleTaskSubmit = async () => {
    if (!onSubmitTask || isSubmitting) return;
    const prof = WORKLOAD_PROFILES[selectedProfile] || WORKLOAD_PROFILES.Medium;
    setIsSubmitting(true);
    setSubmissionFeedback(null);

    try {
      const payload = {
        workload_type: 'Video',
        profile_name: selectedProfile,
        memory_required: prof.memory,
        execution_time: prof.time,
        priority: prof.priority,
        description: prof.desc,
      };
      const res = await onSubmitTask(payload);
      if (res && res.task_id) {
        const dest = res.decision?.selected_gpu || (res.decision?.selected_gpu_index !== undefined ? `GPU ${res.decision.selected_gpu_index + 1}` : 'In Queue');
        setSubmissionFeedback({
          type: 'success',
          msg: `✓ Ingested ${res.task_id} (${prof.memory} GB) ➔ ${dest} [${res.status || 'SCHEDULED'}]`,
        });
      } else {
        setSubmissionFeedback({ type: 'success', msg: `✓ Task submitted to scheduler` });
      }
    } catch (err) {
      setSubmissionFeedback({ type: 'error', msg: `Failed: ${err.message || 'Submission error'}` });
    } finally {
      setIsSubmitting(false);
      setTimeout(() => setSubmissionFeedback(null), 8000);
    }
  };

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Bit 8 Header & View Sub-Navigation */}
      <div className="glass-panel p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-3 border border-cyan-500/30 shadow-[0_0_30px_rgba(6,182,212,0.1)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">🌟</span>
            <h2 className="text-base sm:text-lg font-black tracking-tight text-white uppercase">
              Phase 8: Frontend Dashboard & Dynamic Visualization Suite
            </h2>
            <span className="text-[10px] bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-mono font-bold px-2.5 py-0.5 rounded-full shadow-md shadow-cyan-500/20">
              5 DYNAMIC FEATURES ACTIVE
            </span>
          </div>
          <p className="text-xs text-slate-400 font-medium mt-0.5">
            Full-stack GPU telemetry, task progression pipeline, multi-agent bidding, risk engine, and execution timeline
          </p>
        </div>

        {/* Feature Sub-Navigation */}
        <div className="flex items-center gap-1.5 flex-wrap text-xs font-mono">
          {[
            ['ALL', '🌟 All 5 Features'],
            ['GPU', '🖥️ GPU Dashboard'],
            ['TASK', '🔄 Task Graphs'],
            ['RISK', '🛡️ Risk Engine'],
            ['MARL', '🧠 MARL Decisions'],
            ['EXEC', '⚡ Execution Monitor'],
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setActiveSection(key)}
              className={`px-3 py-1.5 rounded-xl border font-bold transition-all duration-200 ${
                activeSection === key
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white border-cyan-400 shadow-md shadow-cyan-500/30'
                  : 'bg-slate-900/80 text-slate-400 border-white/10 hover:text-white hover:border-white/20'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* ── REAL-TIME SIMULATION & REAL WORKLOAD DISPATCH CONTROL BAR ────────── */}
      <div className="glass-panel p-4 border border-cyan-500/20 rounded-2xl bg-slate-950/80 shadow-lg space-y-3">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-3">
          {/* Simulation Loop State & Step Buttons */}
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className={`px-2.5 py-1 rounded-xl text-xs font-mono font-bold flex items-center gap-2 border ${
              isRunning
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
            }`}>
              <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
              {isRunning ? 'SIMULATION RUNNING' : 'SIMULATION PAUSED'}
            </span>

            {onStartSimulation && !isRunning && (
              <button
                onClick={onStartSimulation}
                className="bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-500/40 text-emerald-300 font-bold px-3 py-1.5 rounded-xl text-xs flex items-center gap-1.5 transition-all shadow-md shadow-emerald-900/20"
              >
                <span>▷</span> Start Simulation
              </button>
            )}

            {onPauseSimulation && isRunning && (
              <button
                onClick={onPauseSimulation}
                className="bg-amber-950/80 hover:bg-amber-900 border border-amber-500/40 text-amber-300 font-bold px-3 py-1.5 rounded-xl text-xs flex items-center gap-1.5 transition-all shadow-md shadow-amber-900/20"
              >
                <span>⏸</span> Pause
              </button>
            )}

            {onStepSimulation && (
              <button
                onClick={onStepSimulation}
                className="bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-500/40 text-cyan-300 font-bold px-3 py-1.5 rounded-xl text-xs flex items-center gap-1.5 transition-all"
                title="Advance simulation by 1 step to execute queued tasks"
              >
                <span>⏩</span> Single Step
              </button>
            )}

            {onResetSimulation && (
              <button
                onClick={onResetSimulation}
                className="bg-slate-900 hover:bg-slate-800 border border-white/10 text-slate-300 font-medium px-2.5 py-1.5 rounded-xl text-xs flex items-center gap-1 transition-all"
                title="Reset environment and clear queues"
              >
                <span>⟳</span> Reset
              </button>
            )}

            {onRefresh && (
              <button
                onClick={onRefresh}
                className="bg-slate-900 hover:bg-slate-800 border border-white/10 text-slate-300 font-medium px-2.5 py-1.5 rounded-xl text-xs flex items-center gap-1 transition-all"
                title="Fetch latest REST snapshot"
              >
                <span>🔄</span> Sync
              </button>
            )}
          </div>

          {/* Real Task Ingestion Dispatcher */}
          <div className="flex items-center gap-2 flex-wrap w-full lg:w-auto justify-end">
            <span className="text-[11px] text-slate-400 font-mono">Workload:</span>
            <div className="bg-slate-900 border border-white/10 rounded-xl p-1 flex gap-1 text-[11px] font-mono">
              {Object.keys(WORKLOAD_PROFILES).map((profKey) => {
                const isCur = selectedProfile === profKey;
                return (
                  <button
                    key={profKey}
                    onClick={() => setSelectedProfile(profKey)}
                    className={`px-2.5 py-1 rounded-lg font-bold transition-all ${
                      isCur
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {profKey}
                  </button>
                );
              })}
            </div>

            <button
              onClick={handleTaskSubmit}
              disabled={isSubmitting}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold px-3.5 py-1.5 rounded-xl text-xs font-mono transition-all shadow-md shadow-cyan-600/30 flex items-center gap-1.5"
            >
              <span>{isSubmitting ? '⏳' : '⚡'}</span>
              <span>{isSubmitting ? 'Scheduling...' : 'Submit Real Video Task'}</span>
            </button>
          </div>
        </div>

        {/* Real-time Feedback Banner */}
        {submissionFeedback && (
          <div className={`p-2.5 rounded-xl border text-xs font-mono flex items-center justify-between animate-fadeIn ${
            submissionFeedback.type === 'error'
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
          }`}>
            <span>{submissionFeedback.msg}</span>
            <button onClick={() => setSubmissionFeedback(null)} className="text-slate-400 hover:text-white">✕</button>
          </div>
        )}
      </div>

      {/* Feature 1: GPU Dashboard */}
      {(activeSection === 'ALL' || activeSection === 'GPU') && (
        <GPUDashboard
          gpus={gpus}
          selectedGpuId={selectedGpuId}
          onSelectGpu={onSelectGpu}
          lastUpdated={lastUpdated}
          isDisconnected={isDisconnected}
        />
      )}

      {/* Feature 2: Dynamic Task Graphs */}
      {(activeSection === 'ALL' || activeSection === 'TASK') && (
        <DynamicTaskGraph
          tasks={tasks}
          videoQueue={videoQueue}
        />
      )}

      {/* Feature 3: Risk Visualization */}
      {(activeSection === 'ALL' || activeSection === 'RISK') && (
        <RiskVisualization
          gpus={gpus}
          riskSummary={riskSummary}
          riskHistory={riskHistory}
          selectedGpuId={selectedGpuId}
          onSelectGpu={onSelectGpu}
        />
      )}

      {/* Feature 4: MARL Decision Visualization */}
      {(activeSection === 'ALL' || activeSection === 'MARL') && (
        <MARLDecisionVisualizer
          decision={marlDecision}
          decisionHistory={decisionHistory}
        />
      )}

      {/* Feature 5: Execution Visualization */}
      {(activeSection === 'ALL' || activeSection === 'EXEC') && (
        <ExecutionVisualizer
          gpus={gpus}
          events={events}
          metrics={metrics}
          tasks={tasks}
        />
      )}
    </div>
  );
}
