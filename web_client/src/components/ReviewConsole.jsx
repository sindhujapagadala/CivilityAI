import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';

export default function ReviewConsole() {
  const [pendingItems, setPendingItems] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [actionInProgress, setActionInProgress] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState(null);

  const fetchQueue = async () => {
    setIsLoading(true);
    try {
      const items = await apiService.getPendingReviews(50, 0);
      setPendingItems(items);
      if (items.length > 0 && !selectedRecord) {
        setSelectedRecord(items[0]);
      }
    } catch (err) {
      console.error('Failed to load pending reviews:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, []);

  const handleAction = async (recordId, action) => {
    setActionInProgress(true);
    try {
      await apiService.submitReviewAction(recordId, action, 'moderator_lead');
      setFeedbackMessage({ type: 'success', text: `Action '${action}' applied successfully to message.` });
      // Remove item from local list
      const updated = pendingItems.filter((item) => item.record_id !== recordId);
      setPendingItems(updated);
      setSelectedRecord(updated.length > 0 ? updated[0] : null);
    } catch (err) {
      setFeedbackMessage({ type: 'error', text: err.message || 'Action submission failed.' });
    } finally {
      setActionInProgress(false);
      setTimeout(() => setFeedbackMessage(null), 4000);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            Human Moderation & Review Console
          </h2>
          <p className="text-xs text-slate-400">
            Audit high-risk messages flagged by automated policy boundaries.
          </p>
        </div>
        <button
          onClick={fetchQueue}
          disabled={isLoading}
          className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition flex items-center gap-1.5"
        >
          <span>🔄</span>
          <span>Refresh Queue ({pendingItems.length})</span>
        </button>
      </div>

      {feedbackMessage && (
        <div
          className={`p-3 rounded-lg text-xs font-semibold flex items-center gap-2 ${
            feedbackMessage.type === 'success'
              ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300'
              : 'bg-rose-500/10 border border-rose-500/30 text-rose-300'
          }`}
        >
          <span>{feedbackMessage.type === 'success' ? '✓' : '⚠'}</span>
          <span>{feedbackMessage.text}</span>
        </div>
      )}

      {/* Main Grid: Queue Table + Inspector Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Review Queue Table (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Pending Adjudication Queue
            </span>
            <span className="text-xs font-mono text-indigo-400">
              {pendingItems.length} awaiting action
            </span>
          </div>

          {isLoading ? (
            <div className="p-12 text-center text-slate-500 text-xs">
              Loading flagged records...
            </div>
          ) : pendingItems.length === 0 ? (
            <div className="p-12 text-center text-slate-500 text-xs space-y-2">
              <span className="text-2xl">🎉</span>
              <p>Queue clear! No items currently flagged for human review.</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-800 max-h-[600px] overflow-y-auto">
              {pendingItems.map((item) => {
                const isSelected = selectedRecord?.record_id === item.record_id;
                const riskPct = Math.round(item.safety_risk_index * 100);

                return (
                  <div
                    key={item.record_id}
                    onClick={() => setSelectedRecord(item)}
                    className={`p-4 cursor-pointer transition-all hover:bg-slate-800/50 ${
                      isSelected ? 'bg-indigo-950/40 border-l-4 border-indigo-500' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${
                            item.moderation_action === 'ESCALATE'
                              ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                              : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                          }`}
                        >
                          {item.moderation_action}
                        </span>
                        <span className="text-xs font-semibold text-slate-300 capitalize">
                          {item.primary_category.replace('_', ' ')}
                        </span>
                      </div>
                      <span className="text-xs font-mono font-bold text-slate-200">
                        Risk: {riskPct}%
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 line-clamp-2 leading-relaxed">
                      "{item.message_body}"
                    </p>

                    <div className="mt-2 text-[10px] text-slate-500 font-mono">
                      Submitted: {new Date(item.submitted_at).toLocaleTimeString()}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Record Detail Inspector & Resolution Controls (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg flex flex-col justify-between">
          {selectedRecord ? (
            <div className="space-y-6">
              <div>
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span className="font-bold uppercase tracking-wider">Record Inspector</span>
                  <span className="font-mono text-[11px]">{selectedRecord.record_id.slice(0, 8)}...</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-sm text-slate-100 leading-relaxed font-sans">
                  "{selectedRecord.message_body}"
                </div>
              </div>

              {/* Multi-category score meters */}
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3">
                  Safety Category Breakdown
                </h4>
                <div className="space-y-2.5">
                  {Object.entries(selectedRecord.category_scores).map(([category, score]) => {
                    const pct = Math.round(score * 100);
                    return (
                      <div key={category} className="text-xs space-y-1">
                        <div className="flex justify-between text-slate-300">
                          <span className="capitalize">{category.replace('_', ' ')}</span>
                          <span className="font-mono font-bold">{pct}%</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-1.5 rounded-full ${
                              pct > 50 ? 'bg-rose-500' : pct > 25 ? 'bg-amber-500' : 'bg-indigo-500'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-4 border-t border-slate-800 space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                  Adjudication Action
                </span>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => handleAction(selectedRecord.record_id, 'ALLOW')}
                    disabled={actionInProgress}
                    className="px-3 py-2 rounded-xl bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 font-bold text-xs transition disabled:opacity-50"
                  >
                    ✓ ALLOW
                  </button>
                  <button
                    onClick={() => handleAction(selectedRecord.record_id, 'REMOVE')}
                    disabled={actionInProgress}
                    className="px-3 py-2 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 font-bold text-xs transition disabled:opacity-50"
                  >
                    ✕ REMOVE
                  </button>
                  <button
                    onClick={() => handleAction(selectedRecord.record_id, 'ESCALATE')}
                    disabled={actionInProgress}
                    className="px-3 py-2 rounded-xl bg-amber-600/20 hover:bg-amber-600/30 text-amber-300 border border-amber-500/30 font-bold text-xs transition disabled:opacity-50"
                  >
                    🚨 ESCALATE
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-xs">
              Select a message from the queue to inspect details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
