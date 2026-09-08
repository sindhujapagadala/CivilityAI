import React, { useEffect, useState } from 'react';
import NavigationHeader from './components/NavigationHeader';
import ContentAnalyzer from './components/ContentAnalyzer';
import ReviewConsole from './components/ReviewConsole';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import BatchProcessing from './components/BatchProcessing';
import { apiService } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const health = await apiService.checkHealth();
        setSystemStatus(health);
      } catch (err) {
        console.warn('Backend service offline or initializing:', err.message);
        setSystemStatus({ status: 'offline', engine_version: 'CivilityAI-v1.0.0 (Offline)' });
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Navigation Bar */}
      <NavigationHeader
        activeTab={activeTab}
        onTabChange={setActiveTab}
        systemStatus={systemStatus}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'analyzer' && <ContentAnalyzer />}
        {activeTab === 'review' && <ReviewConsole />}
        {activeTab === 'analytics' && <AnalyticsDashboard />}
        {activeTab === 'batch' && <BatchProcessing />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>CivilityAI — Intelligent Toxic Content Detection & Moderation Platform</span>
          <span className="font-mono text-[11px] text-slate-600">
            PyTorch • DistilBERT • Multi-Label BCEWithLogitsLoss • FastAPI
          </span>
        </div>
      </footer>
    </div>
  );
}
