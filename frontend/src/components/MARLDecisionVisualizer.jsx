import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function MARLDecisionVisualizer({
  decision = null,
  decisionHistory = [],
}) {
  const candidateScores = useMemo(() => {
    if (!decision?.candidates || !Array.isArray(decision.candidates)) return [];
    return decision.candidates.map((c) => ({
      ...c,
      displayScore: c.eligible ? Math.max(0, c.bid_score) : 0,
      isSelected: decision.selected_gpu === c.label || decision.selected_gpu_id === c.gpu_id,
    }));
  }, [decision]);

  return (
    <section className="glass-panel p-4 space-y-4 border border-white/10">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-emerald-400 text-lg">🧠</span>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              MARL Cooperative Decision & Candidate Bidding Decomposition
            </h3>
            <p className="text-[11px] text-slate-400">
              Multi-agent Q-learning bidding, risk penalties, and eligibility constraints
            </p>
          </div>
        </div>
        {decision && (
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-400">Decision ID:</span>
            <span className="bg-slate-900 border border-white/10 text-cyan-400 px-2 py-0.5 rounded-lg">
              {decision.decision_id || `task-${decision.task_id}`}
            </span>
          </div>
        )}
      </div>

      {decision ? (
        <div className="space-y-4">
          {/* Selected GPU Hero & Task Details */}
          <div className="bg-slate-950/80 border border-emerald-500/30 rounded-2xl p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs text-slate-400 uppercase font-mono">Selected Partition:</span>
                <span className="text-2xl font-black text-emerald-400 font-mono">
                  {decision.selected_gpu || (decision.selected_gpu_id ? `GPU ${decision.selected_gpu_id}` : 'None (Rejected)')}
                </span>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold font-mono ${
                  decision.status === 'assigned'
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                }`}>
                  {decision.status?.toUpperCase() || 'EVALUATED'}
                </span>
              </div>
              <p className="text-xs text-slate-300 font-mono">
                Task ID: <strong className="text-cyan-400">{decision.task_id}</strong> · Required VRAM: <strong className="text-white">{decision.task_memory} GB</strong> · Execution Time: <strong className="text-white">{decision.task_time}s</strong>
              </p>
              {decision.reason && (
                <p className="text-[11px] text-slate-400 italic">
                  Recorded Reason: "{decision.reason}"
                </p>
              )}
            </div>

            <div className="flex md:flex-col gap-3 md:gap-1 text-left md:text-right font-mono text-xs">
              <div>
                <span className="text-[10px] text-slate-500 uppercase">Winning Bid Score: </span>
                <span className="text-emerald-400 font-black text-base">
                  {typeof decision.decision_score === 'number' ? decision.decision_score.toFixed(4) : '—'}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase">Risk / EFC: </span>
                <span className="text-amber-400 font-bold">{(decision.risk_score ?? 0).toFixed(4)}</span>
                <span className="text-slate-500"> / </span>
                <span className="text-purple-400 font-bold">{(decision.expected_failure_cost ?? 0).toFixed(4)}</span>
              </div>
              {decision.reward !== null && decision.reward !== undefined && (
                <div>
                  <span className="text-[10px] text-slate-500 uppercase">RL Reward: </span>
                  <span className="text-cyan-400 font-bold">{Number(decision.reward).toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Candidate GPU Scores Ranked Bar Chart */}
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 lg:col-span-5 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-2">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                Candidate Bid Scores Comparison
              </span>
              {candidateScores.length > 0 ? (
                <div className="w-full h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={candidateScores} layout="vertical" margin={{ top: 5, right: 25, left: 20, bottom: 5 }}>
                      <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
                      <YAxis type="category" dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 'bold' }} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const c = payload[0]?.payload;
                          return (
                            <div className="bg-slate-900 border border-cyan-500/40 p-2.5 rounded-xl text-xs font-mono shadow-xl space-y-1">
                              <p className="font-bold text-white border-b border-white/10 pb-1 flex justify-between">
                                <span>{c.label}</span>
                                <span className={c.eligible ? 'text-emerald-400' : 'text-rose-400'}>
                                  {c.eligible ? 'ELIGIBLE' : c.ineligible_reason || 'INELIGIBLE'}
                                </span>
                              </p>
                              <div className="text-[11px] space-y-0.5 pt-0.5">
                                <div className="text-slate-300">Bid: <strong className="text-cyan-300">{c.bid_score > -50 ? c.bid_score.toFixed(4) : 'N/A'}</strong></div>
                                <div className="text-rose-400">Risk Penalty: -{c.risk_penalty?.toFixed(4) ?? '0.0000'}</div>
                                <div className="text-amber-400">EFC Penalty: -{c.efc_penalty?.toFixed(4) ?? '0.0000'}</div>
                                <div className="text-emerald-400">Perf Score: +{c.perf_score?.toFixed(4) ?? '0.0000'}</div>
                                <div className="text-purple-400">Q Contribution: +{c.q_contribution?.toFixed(4) ?? '0.0000'}</div>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Bar dataKey="displayScore" radius={[0, 4, 4, 0]}>
                        {candidateScores.map((entry, index) => (
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
                <p className="text-xs text-slate-500 text-center py-8">No candidate breakdown available.</p>
              )}
            </div>

            {/* Candidate Breakdown Matrix Table */}
            <div className="col-span-12 lg:col-span-7 bg-slate-950/70 border border-white/5 rounded-2xl p-3.5 space-y-2 overflow-x-auto">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                Recorded Scoring Component Breakdown
              </span>
              <table className="w-full text-left font-mono text-[11px]">
                <thead className="bg-slate-900/90 text-slate-400 text-[9px] uppercase border-b border-white/10">
                  <tr>
                    <th className="p-2">Candidate</th>
                    <th className="p-2">Admission</th>
                    <th className="p-2">Perf Score</th>
                    <th className="p-2">Risk Pen</th>
                    <th className="p-2">EFC Pen</th>
                    <th className="p-2">Q-Val</th>
                    <th className="p-2 text-right">Final Bid</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {candidateScores.map((c) => (
                    <tr
                      key={c.gpu_id}
                      className={c.isSelected ? 'bg-emerald-500/10 text-emerald-300 font-bold' : 'text-slate-300'}
                    >
                      <td className="p-2 flex items-center gap-1.5">
                        <span>{c.label}</span>
                        {c.isSelected && <span className="text-emerald-400 text-[10px]">✓</span>}
                      </td>
                      <td className="p-2">
                        <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                          c.eligible ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                        }`}>
                          {c.eligible ? 'ELIGIBLE' : c.ineligible_reason || 'INELIGIBLE'}
                        </span>
                      </td>
                      <td className="p-2 text-cyan-400">+{c.perf_score?.toFixed(4) ?? '0.0000'}</td>
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
          </div>

          {/* Decision History Log */}
          <div className="bg-slate-950/60 border border-white/5 rounded-2xl p-3 space-y-2">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
              Recent Decision History (Latest {decisionHistory.length} Recorded Decisions)
            </span>
            <div className="max-h-36 overflow-y-auto space-y-1 font-mono text-[10px]">
              {decisionHistory.map((d, i) => (
                <div
                  key={d.decision_id || i}
                  className="p-1.5 px-2.5 rounded-lg bg-slate-900/60 border border-white/5 flex justify-between items-center text-slate-300"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-cyan-400 font-bold">{d.task_id}</span>
                    <span className="text-slate-400">
                      Target: <strong className="text-white">{d.selected_gpu || 'Rejected'}</strong>
                    </span>
                    <span className="text-slate-500">
                      Mem: {d.task_memory}G
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-emerald-400 font-bold">
                      Bid: {typeof d.decision_score === 'number' ? d.decision_score.toFixed(3) : '—'}
                    </span>
                    <span className="text-amber-400">
                      Risk: {(d.risk_score ?? 0).toFixed(3)}
                    </span>
                    <span className={`px-1.5 py-0.2 rounded text-[9px] ${
                      d.status === 'assigned' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                    }`}>
                      {d.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12 text-slate-600 text-xs">
          Awaiting MARL scheduling requests...
        </div>
      )}
    </section>
  );
}
