import { motion } from 'framer-motion';
import GlobalRadar from './GlobalRadar';

export interface Theme {
  topic_label: string;
  size: number;
}

export interface GlobalInsightsData {
  total_dreams: number;
  unique_archetypes: number;
  semantic_density: number;
  dominant_tone: string;
  emotion_radar: Record<string, number>;
  top_themes: Theme[];
}

interface Props {
  data: GlobalInsightsData;
  onClose: () => void;
}

export default function GlobalInsights({ data, onClose }: Props) {
  const maxThemeSize = Math.max(...data.top_themes.map(t => t.size), 1);

  return (
    <section
      id="global-insights"
      style={{
        position: 'relative',
        minHeight: '100svh',
        background: 'var(--bg)',
        padding: '80px 24px 120px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: '100px'
      }}
    >
      <button 
        onClick={onClose} 
        style={{ 
          position: 'absolute', 
          top: '32px', 
          right: '48px', 
          color: '#fff', 
          background: 'transparent', 
          border: '1px solid rgba(255,255,255,0.3)', 
          padding: '8px 16px', 
          borderRadius: '4px', 
          cursor: 'pointer', 
          fontFamily: 'var(--font-sans)', 
          textTransform: 'uppercase', 
          letterSpacing: '0.1em', 
          fontSize: '0.7rem' 
        }}>
        Close
      </button>

      {/* Label */}
      <motion.p
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.72rem',
          letterSpacing: '0.22em',
          color: 'var(--accent)',
          textTransform: 'uppercase',
          marginBottom: '12px',
        }}
      >
        Command Center
      </motion.p>

      {/* Header */}
      <motion.h2
        initial={{ opacity: 0, y: 28 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.05 }}
        style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 'clamp(2.4rem, 5vw, 4rem)',
          fontWeight: 400,
          color: 'var(--text)',
          textAlign: 'center',
          letterSpacing: '-0.02em',
          lineHeight: 1.15,
          marginBottom: '56px',
        }}
      >
        Global Insights
      </motion.h2>

      {/* Liminal Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.15 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          width: '100%',
          maxWidth: '960px',
          marginBottom: '40px',
        }}
      >
        <MetricCard label="Analyzed Fragments" value={data.total_dreams.toLocaleString()} />
        <MetricCard label="Unique Archetypes" value={data.unique_archetypes.toLocaleString()} />
        <MetricCard label="Semantic Density" value={data.semantic_density} />
        <MetricCard label="Dominant Tone" value={data.dominant_tone} />
      </motion.div>

      {/* Charts Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
          gap: '20px',
          width: '100%',
          maxWidth: '960px',
          marginBottom: '20px',
        }}
      >
        {/* Emotion Radar */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="glass"
          style={{
            padding: '36px 40px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            background: '#000',
            border: '1px solid rgba(111,111,111,0.3)',
          }}
        >
          <p
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.68rem',
              letterSpacing: '0.2em',
              color: 'var(--accent)',
              textTransform: 'uppercase',
              marginBottom: '28px',
              textAlign: 'center',
            }}
          >
            The Affective Core
          </p>
          <div style={{ width: '100%', maxWidth: '350px' }}>
            <GlobalRadar emotionVector={data.emotion_radar} />
          </div>
        </motion.div>

        {/* Theme Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="glass"
          style={{
            padding: '36px 40px',
            display: 'flex',
            flexDirection: 'column',
            background: '#000',
            border: '1px solid rgba(111,111,111,0.3)',
          }}
        >
          <p
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.68rem',
              letterSpacing: '0.2em',
              color: 'var(--accent)',
              textTransform: 'uppercase',
              marginBottom: '28px',
              textAlign: 'center',
            }}
          >
            Thematic Landscape
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {data.top_themes.map((theme, i) => (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: '#fff' }}>{theme.topic_label}</span>
                  <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.8rem', color: 'rgba(255,255,255,0.6)' }}>{theme.size}</span>
                </div>
                <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(theme.size / maxThemeSize) * 100}%` }}
                    transition={{ duration: 1, delay: 0.5 + (i * 0.1) }}
                    style={{ height: '100%', background: '#6F6F6F', borderRadius: '2px', boxShadow: '0 0 8px #6F6F6F' }}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      className="glass"
      style={{
        background: '#000',
        border: '1px solid rgba(111,111,111,0.3)',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
      }}
    >
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.65rem',
          letterSpacing: '0.15em',
          color: 'rgba(255,255,255,0.5)',
          textTransform: 'uppercase',
          marginBottom: '12px',
        }}
      >
        {label}
      </p>
      <h3
        style={{
          fontFamily: 'var(--font-serif)',
          fontSize: '1.8rem',
          fontWeight: 400,
          color: '#fff',
        }}
      >
        {value}
      </h3>
    </div>
  );
}
