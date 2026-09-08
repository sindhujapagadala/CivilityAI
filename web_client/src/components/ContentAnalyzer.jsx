import React, { useState } from 'react';
import { apiService } from '../services/api';

const CATEGORY_META = [
  { key: 'general_toxicity', label: 'General Toxicity', desc: 'Rude, disrespectful, or toxic speech', color: 'from-amber-500 to-rose-500' },
  { key: 'severe_abuse', label: 'Severe Abuse', desc: 'Extremely aggressive or hateful abuse', color: 'from-rose-600 to-red-600' },
  { key: 'obscene_language', label: 'Obscene Language', desc: 'Vulgarity, profanity, or sexual obscenity', color: 'from-purple-500 to-indigo-500' },
  { key: 'threatening_language', label: 'Threatening Language', desc: 'Direct threats of physical harm or violence', color: 'from-red-600 to-rose-700' },
  { key: 'personal_insult', label: 'Personal Insult', desc: 'Disparaging, demeaning, or attacking an individual', color: 'from-orange-500 to-amber-500' },
  { key: 'identity_attack', label: 'Identity Attack', desc: 'Hate speech targeting race, religion, or gender', color: 'from-pink-600 to-rose-600' },
];

const PRESET_EXAMPLES = [
  { label: 'Civil Discussion', text: 'Thank you for improving this article, the references look very solid and well-researched.' },
  { label: 'Personal Insult', text: 'You are completely incompetent and have no idea what you are talking about.' },
  { label: 'Violent Threat', text: 'I know where you live and I will track you down and break your skull.' },
  { label: 'Obscenity & Hate', text: 'Shut the fuck up and delete your account, you worthless parasite.' },
];

export default function ContentAnalyzer() {
  const [messageText, setMessageText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    if (!messageText.trim()) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const result = await apiService.analyzeMessage(messageText);
      setAnalysisResult(result);
    } catch (err) {
      setErrorMessage(err.message || 'Failed to analyze content.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getActionBadge = (action) => {
    switch (action) {
      case 'ALLOW':
        return {
          bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          icon: '✓',
          title: 'ALLOW',
          sub: 'Content passes automated safety policies.',
        };
      case 'REVIEW':
        return {
          bg: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
          icon: '⚠',
          title: 'HUMAN REVIEW',
          sub: 'Content routed to human moderators for review.',
        };
      case 'ESCALATE':
        return {
          bg: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
          icon: '🚨',
          title: 'ESCALATE',
          sub: 'Severe safety violation. Immediate enforcement action taken.',
        };
      default:
        return {
          bg: 'bg-slate-800 text-slate-300 border-slate-700',
          icon: '•',
          title: action,
          sub: '',
        };
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Title & Introduction */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
          Intelligent Content Safety & Moderation
        </h2>
        <p className="text-slate-400 text-sm max-w-2xl mx-auto">
          Multi-label safety evaluation assessing 6 toxicity dimensions, computing composite Safety Risk Index,
          and enforcing policy governance.
        </p>
      </div>

      {/* Content Input Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <form onSubmit={handleAnalyze} className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Message Text Body
            </label>
            <span className="text-xs text-slate-500 font-mono">
              {messageText.length} characters
            </span>
          </div>

          <div className="relative">
            <textarea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              placeholder="Enter content to analyze..."
              rows={4}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition text-sm leading-relaxed"
            />
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mr-1">
              Sample Prompts:
            </span>
            {PRESET_EXAMPLES.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setMessageText(preset.text);
                }}
                className="text-xs px-2.5 py-1 rounded-md bg-slate-800/70 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/60 transition"
              >
                {preset.label}
              </button>
            ))}
          </div>

          {/* Analyze Action Button */}
          <div className="pt-2 flex items-center justify-end">
            <button
              type="submit"
              disabled={isAnalyzing || !messageText.trim()}
              className="px-6 py-2.5 rounded-xl font-bold text-sm bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-lg shadow-indigo-600/30 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>ANALYZING CONTENT...</span>
                </>
              ) : (
                <>
                  <span>🛡️</span>
                  <span>ANALYZE CONTENT</span>
                </>
              )}
            </button>
          </div>
        </form>

        {errorMessage && (
          <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
            <span>⚠</span>
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {/* Safety Analysis Results Panel */}
      {analysisResult && (
        <div className="space-y-6 animate-fadeIn">
          {/* Header Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Safety Risk Index */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden flex flex-col justify-between">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-bold uppercase tracking-wider">Safety Risk Index</span>
                <span className="font-mono text-indigo-400">Composite</span>
              </div>
              <div className="my-3 flex items-baseline gap-2">
                <span className="text-4xl font-black tracking-tight text-white font-mono">
                  {Math.round(analysisResult.safety_risk_index * 100)}%
                </span>
                <span className="text-xs text-slate-400">risk probability</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${
                    analysisResult.safety_risk_index >= 0.75
                      ? 'bg-rose-500'
                      : analysisResult.safety_risk_index >= 0.30
                      ? 'bg-amber-500'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(analysisResult.safety_risk_index * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* Moderation Action */}
            {(() => {
              const actionBadge = getActionBadge(analysisResult.moderation_action);
              return (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                  <div className="text-xs text-slate-400 font-bold uppercase tracking-wider">
                    Moderation Action
                  </div>
                  <div className="my-2">
                    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border text-sm font-extrabold ${actionBadge.bg}`}>
                      <span>{actionBadge.icon}</span>
                      <span>{actionBadge.title}</span>
                    </div>
                    <p className="text-xs text-slate-400 mt-2">{actionBadge.sub}</p>
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono">
                    Triggered Violations: {analysisResult.triggered_categories.length}
                  </div>
                </div>
              );
            })()}

            {/* Inference Telemetry */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
              <div className="text-xs text-slate-400 font-bold uppercase tracking-wider">
                Inference Telemetry
              </div>
              <div className="space-y-1.5 my-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Processing Latency:</span>
                  <span className="font-mono text-slate-200 font-bold">{analysisResult.processing_time_ms} ms</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Engine Version:</span>
                  <span className="font-mono text-slate-200">{analysisResult.engine_version}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Record ID:</span>
                  <span className="font-mono text-slate-400 text-[10px] truncate max-w-[120px]">{analysisResult.record_id || 'Ephemeral'}</span>
                </div>
              </div>
              <div className="text-[10px] text-slate-500">
                Multi-Label Sigmoid Activation
              </div>
            </div>
          </div>

          {/* 6 Categories Detailed Progress Bars */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center justify-between">
              <span>Safety Category Breakdown</span>
              <span className="text-xs text-slate-500 font-normal">Independent Multi-Label Probabilities</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {CATEGORY_META.map((category) => {
                const score = analysisResult.category_scores[category.key] || 0.0;
                const percentage = Math.round(score * 100);
                const isTriggered = analysisResult.triggered_categories.includes(category.key);

                return (
                  <div
                    key={category.key}
                    className={`p-3.5 rounded-xl border transition-all ${
                      isTriggered
                        ? 'bg-slate-800/70 border-rose-500/40 ring-1 ring-rose-500/20'
                        : 'bg-slate-950/60 border-slate-800/80'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="font-bold text-xs text-slate-200">{category.label}</span>
                          {isTriggered && (
                            <span className="text-[9px] uppercase font-bold px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                              FLAGGED
                            </span>
                          )}
                        </div>
                        <p className="text-[10px] text-slate-500">{category.desc}</p>
                      </div>
                      <span className={`font-mono text-sm font-bold ${percentage > 50 ? 'text-rose-400' : percentage > 25 ? 'text-amber-400' : 'text-slate-300'}`}>
                        {percentage}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-2 rounded-full bg-gradient-to-r ${category.color} transition-all duration-500`}
                        style={{ width: `${Math.min(percentage, 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
