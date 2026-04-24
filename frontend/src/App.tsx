import { useState, useRef } from 'react';
import HeroSection from './components/HeroSection';
import DreamChamber from './components/DreamChamber';
import ProjectionDashboard from './components/ProjectionDashboard';
import GlobalInsights, { type GlobalInsightsData } from './components/GlobalInsights';

interface AnalysisResult {
  Topic_Cluster: string;
  Dominant_Emotion: string;
  Key_Entities: string[];
  Semantic_Relation: { Agent: string; Action: string; Target: string }[];
  Emotion_Vector: Record<string, number>;
  Global_Stat: string;
}

type AppState = 'hero' | 'chamber' | 'loading' | 'projection' | 'global_insights';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export default function App() {
  const [appState, setAppState] = useState<AppState>('hero');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [globalData, setGlobalData] = useState<GlobalInsightsData | null>(null);

  const chamberRef = useRef<HTMLDivElement>(null);
  const projectionRef = useRef<HTMLDivElement>(null);
  const globalRef = useRef<HTMLDivElement>(null);

  const viewGlobalInsights = async () => {
    window.scrollTo({ top: 0, behavior: 'smooth' }); // Scroll to top immediately
    setAppState('loading'); 
    try {
      if (!globalData) {
        const res = await fetch(`${API_BASE}/global-insights`);
        if (!res.ok) throw new Error("Failed to load global insights");
        const data = await res.json();
        setGlobalData(data);
      }
      setAppState('global_insights');
      setTimeout(() => {
        globalRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    } catch (err) {
      setError((err as Error).message);
      setAppState('hero');
    }
  };

  const scrollToChamber = () => {
    setAppState('chamber');
    setTimeout(() => {
      chamberRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 50);
  };

  const handleSubmit = async (dreamText: string) => {
    setAppState('loading');
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dream_text: dreamText }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }

      const data: AnalysisResult = await res.json();
      setResult(data);
      setAppState('projection');
      setTimeout(() => {
        projectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    } catch (err) {
      setError((err as Error).message);
      setAppState('chamber');
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setAppState('chamber');
    setTimeout(() => {
      chamberRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 50);
  };

  const isLoading = appState === 'loading';

  return (
    <main>
      {/* Hero — always rendered for scroll reference */}
      <HeroSection onExplore={scrollToChamber} onGlobalInsights={viewGlobalInsights} />

      {/* Chamber — mounted once user scrolls down */}
      {(appState === 'chamber' || appState === 'loading' || appState === 'projection') && (
        <div ref={chamberRef}>
          <DreamChamber onSubmit={handleSubmit} isLoading={isLoading} />
        </div>
      )}

      {/* Error toast */}
      {error && (
        <div
          role="alert"
          style={{
            position: 'fixed',
            bottom: '32px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#000',
            color: '#fff',
            fontFamily: 'var(--font-sans)',
            fontSize: '0.85rem',
            padding: '14px 28px',
            borderRadius: '12px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
            zIndex: 999,
            maxWidth: '480px',
            textAlign: 'center',
          }}
        >
          ⚠ {error}
          <button
            onClick={() => setError(null)}
            style={{
              marginLeft: '16px',
              background: 'rgba(255,255,255,0.15)',
              border: 'none',
              color: '#fff',
              borderRadius: '99px',
              padding: '2px 10px',
              cursor: 'pointer',
              fontSize: '0.78rem',
            }}
          >
            dismiss
          </button>
        </div>
      )}

      {/* Projection dashboard */}
      {appState === 'projection' && result && (
        <div ref={projectionRef}>
          <ProjectionDashboard result={result} onReset={handleReset} />
        </div>
      )}

      {/* Global insights dashboard */}
      {appState === 'global_insights' && globalData && (
        <div ref={globalRef}>
          <GlobalInsights data={globalData} onClose={() => { setAppState('hero'); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
        </div>
      )}
    </main>
  );
}
