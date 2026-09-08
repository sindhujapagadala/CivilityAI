import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';

export default function AnalyticsDashboard() {
  const [summary, setSummary] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getAnalyticsSummary();
      setSummary(data);
    } catch (err) {
      console.error('Failed to load analytics:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto p-12 text-center text-slate-500 text-xs">
        Loading analytics metrics...
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="max-w-7xl mx-auto p-12 text-center text-slate-500 text-xs">
        Unable to load analytics telemetry.
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Platform Safety & Telemetry Analytics
          </h2>
          <p className="text-xs text-slate-400">
            Real-time insights across content volume, policy distribution, and inference latency.
          </p>
        </div>
        <button
          onClick={fetchAnalytics}
          className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition"
        >
          🔄 Refresh Telemetry
        </button>
      </div>

      {/* 4 Primary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Content Analyzed */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Total Analyzed
          </div>
          <div className="mt-2 text-3xl font-extrabold text-white font-mono">
            {summary.total_analyzed.toLocaleString()}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            All user submissions
          </div>
        </div>

        {/* Allowed Rate */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow">
          <div className="text-xs font-bold uppercase tracking-wider text-emerald-400">
            Allowed Content
          </div>
          <div className="mt-2 text-3xl font-extrabold text-white font-mono">
            {summary.allowed_percentage}%
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            {summary.allowed_count.toLocaleString()} civil messages
          </div>
        </div>

        {/* Review Queue Rate */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow">
          <div className="text-xs font-bold uppercase tracking-wider text-amber-400">
            Review Queue
          </div>
          <div className="mt-2 text-3xl font-extrabold text-white font-mono">
            {summary.review_percentage}%
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            {summary.review_count.toLocaleString()} routed to human review
          </div>
        </div>

        {/* Escalated Rate */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow">
          <div className="text-xs font-bold uppercase tracking-wider text-rose-400">
            Escalated Violations
          </div>
          <div className="mt-2 text-3xl font-extrabold text-white font-mono">
            {summary.escalated_percentage}%
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            {summary.escalated_count.toLocaleString()} severe violations
          </div>
        </div>
      </div>

      {/* Second Row: Latency Benchmarks & Category Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Category Distribution (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
            Violation Distribution by Safety Category
          </h3>

          <div className="space-y-4">
            {Object.entries(summary.category_distribution).map(([cat, count]) => {
              const maxCount = Math.max(...Object.values(summary.category_distribution), 1);
              const barWidth = Math.round((count / maxCount) * 100);

              return (
                <div key={cat} className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span className="capitalize font-medium">{cat.replace('_', ' ')}</span>
                    <span className="font-mono font-bold">{count} occurrences</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800/80">
                    <div
                      className="h-2 rounded-full bg-indigo-500 transition-all duration-500"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Latency Telemetry (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
              Inference Latency Metrics
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400 block font-medium">Mean Latency</span>
                <span className="text-2xl font-bold font-mono text-indigo-400 mt-1 block">
                  {summary.average_processing_time_ms} ms
                </span>
                <span className="text-[10px] text-slate-500">Average execution</span>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                <span className="text-[11px] text-slate-400 block font-medium">P95 Latency</span>
                <span className="text-2xl font-bold font-mono text-violet-400 mt-1 block">
                  {summary.p95_processing_time_ms} ms
                </span>
                <span className="text-[10px] text-slate-500">95th percentile</span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-500/20 text-xs text-indigo-300 leading-relaxed mt-4">
            <span className="font-bold">SLO Target Status:</span> Single-message inference maintains &lt;50ms response latency, satisfying real-time production requirements.
          </div>
        </div>
      </div>

      {/* Recent High-Risk Content Log */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
          Recent High-Risk Content Log
        </h3>

        {summary.recent_high_risk_content.length === 0 ? (
          <div className="text-center py-6 text-slate-500 text-xs">
            No high-risk records logged yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="pb-3 font-semibold">Time</th>
                  <th className="pb-3 font-semibold">Risk Index</th>
                  <th className="pb-3 font-semibold">Action</th>
                  <th className="pb-3 font-semibold">Message Preview</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {summary.recent_high_risk_content.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition">
                    <td className="py-3 font-mono text-slate-400">
                      {new Date(item.assessed_at).toLocaleTimeString()}
                    </td>
                    <td className="py-3 font-mono font-bold text-rose-400">
                      {Math.round(item.safety_risk_index * 100)}%
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                        {item.moderation_action}
                      </span>
                    </td>
                    <td className="py-3 max-w-md truncate font-sans text-slate-200">
                      "{item.message_body}"
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
