import React, { useState } from 'react';
import { apiService } from '../services/api';

export default function BatchProcessing() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchMetrics, setBatchMetrics] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setBatchMetrics(null);
      setErrorMessage(null);
    }
  };

  const handleProcessBatch = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const response = await apiService.uploadBatchCsv(selectedFile);
      setBatchMetrics(response);
    } catch (err) {
      setErrorMessage(err.message || 'Batch inference failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          High-Throughput Batch Moderation
        </h2>
        <p className="text-xs text-slate-400">
          Upload bulk message CSVs for parallel safety scoring, policy classification, and dataset enrichment.
        </p>
      </div>

      {/* Upload Box */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500/60 rounded-xl p-8 text-center transition-all bg-slate-950/40">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            id="batch-csv-upload"
          />
          <label htmlFor="batch-csv-upload" className="cursor-pointer space-y-2 block">
            <span className="text-3xl block">📄</span>
            <div className="text-sm font-semibold text-slate-200">
              {selectedFile ? selectedFile.name : 'Click to select CSV or drag file here'}
            </div>
            <p className="text-xs text-slate-500">
              Expected format: CSV containing <code className="text-indigo-400 font-mono">message_body</code> or <code className="text-indigo-400 font-mono">comment_text</code>
            </p>
          </label>
        </div>

        {selectedFile && (
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">
              File size: {(selectedFile.size / 1024).toFixed(1)} KB
            </span>
            <button
              onClick={handleProcessBatch}
              disabled={isProcessing}
              className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-xs text-white shadow-lg shadow-indigo-600/30 transition disabled:opacity-50 flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Processing Batch...</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Run Batch Inference</span>
                </>
              )}
            </button>
          </div>
        )}

        {errorMessage && (
          <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
            {errorMessage}
          </div>
        )}
      </div>

      {/* Batch Processing Telemetry Cards */}
      {batchMetrics && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Total Messages
              </span>
              <span className="text-2xl font-bold font-mono text-white mt-1 block">
                {batchMetrics.total_messages.toLocaleString()}
              </span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Processing Duration
              </span>
              <span className="text-2xl font-bold font-mono text-indigo-400 mt-1 block">
                {batchMetrics.processing_duration_seconds}s
              </span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Throughput Rate
              </span>
              <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
                {batchMetrics.throughput_rate_msgs_per_sec} msg/s
              </span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Average Latency
              </span>
              <span className="text-2xl font-bold font-mono text-violet-400 mt-1 block">
                {batchMetrics.average_latency_ms} ms
              </span>
            </div>
          </div>

          {/* Sample Preview Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4">
              Enriched Output Preview (First 10 Records)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="pb-3">Action</th>
                    <th className="pb-3">Risk Index</th>
                    <th className="pb-3">General Tox</th>
                    <th className="pb-3">Severe</th>
                    <th className="pb-3">Threat</th>
                    <th className="pb-3">Insult</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono">
                  {batchMetrics.batch_sample_preview.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30">
                      <td className="py-2.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          row.moderation_action === 'ESCALATE' ? 'bg-rose-500/20 text-rose-300' :
                          row.moderation_action === 'REVIEW' ? 'bg-amber-500/20 text-amber-300' :
                          'bg-emerald-500/20 text-emerald-300'
                        }`}>
                          {row.moderation_action}
                        </span>
                      </td>
                      <td className="py-2.5 font-bold text-white">
                        {Math.round(row.safety_risk_index * 100)}%
                      </td>
                      <td className="py-2.5">{(row.score_general_toxicity * 100).toFixed(0)}%</td>
                      <td className="py-2.5">{(row.score_severe_abuse * 100).toFixed(0)}%</td>
                      <td className="py-2.5">{(row.score_threatening_language * 100).toFixed(0)}%</td>
                      <td className="py-2.5">{(row.score_personal_insult * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
