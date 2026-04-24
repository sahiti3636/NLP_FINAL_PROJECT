import { motion } from 'framer-motion';
import RadarChart from './RadarChart';


interface AnalysisResult {
  Topic_Cluster: string;
  Dominant_Emotion: string;
  Key_Entities: string[];
  Semantic_Relation: { Agent: string; Action: string; Target: string }[];
  Emotion_Vector: Record<string, number>;
  Global_Stat: string;
}

interface ProjectionDashboardProps {
  result: AnalysisResult;
  onReset: () => void;
}

const EMOTION_PALETTE: Record<string, { bg: string; text: string }> = {
  fear:         { bg: 'rgba(107, 123, 255, 0.1)',  text: '#5a6aff' },
  sadness:      { bg: 'rgba(90, 143, 255, 0.1)',   text: '#4a7aff' },
  anger:        { bg: 'rgba(255, 107, 107, 0.1)',  text: '#e04040' },
  joy:          { bg: 'rgba(255, 209, 102, 0.12)', text: '#b8860b' },
  anticipation: { bg: 'rgba(6, 214, 160, 0.1)',    text: '#059669' },
  disgust:      { bg: 'rgba(155, 114, 207, 0.1)',  text: '#7c3aed' },
  surprise:     { bg: 'rgba(244, 162, 97, 0.1)',   text: '#d97706' },
  trust:        { bg: 'rgba(67, 197, 158, 0.1)',   text: '#0d9488' },
};

export default function ProjectionDashboard({ result, onReset }: ProjectionDashboardProps) {
  const emotion     = result.Dominant_Emotion.toLowerCase();
  const emotionStyle = EMOTION_PALETTE[emotion] ?? { bg: 'rgba(0,0,0,0.05)', text: '#000' };

  return (
    <section
      id="projection"
      style={{
        minHeight: '100svh',
        background: 'var(--bg)',
        padding: '80px 24px 120px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
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
        The Projection
      </motion.p>

      {/* Topic cluster — large serif header */}
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
          marginBottom: '10px',
          maxWidth: '700px',
        }}
      >
        {result.Topic_Cluster}
      </motion.h2>

      {/* Dominant emotion pill */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.15 }}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          background: emotionStyle.bg,
          borderRadius: '99px',
          padding: '6px 18px',
          marginBottom: '56px',
        }}
      >
        <span
          style={{
            width: '7px',
            height: '7px',
            borderRadius: '50%',
            background: emotionStyle.text,
            display: 'inline-block',
          }}
        />
        <span
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.8rem',
            fontWeight: 600,
            color: emotionStyle.text,
            letterSpacing: '0.06em',
            textTransform: 'capitalize',
          }}
        >
          {result.Dominant_Emotion}
        </span>
      </motion.div>

      {/* Grid: top row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px',
          width: '100%',
          maxWidth: '960px',
          marginBottom: '20px',
        }}
      >
        {/* NRC Bar Chart card */}
        <Card delay={0.2} label="NRC Emotion Spectrum">
          <RadarChart
            emotionVector={result.Emotion_Vector}
            dominantEmotion={result.Dominant_Emotion}
          />
        </Card>

        {/* Entities card */}
        <Card delay={0.3} label="Key Entities">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center', padding: '10px 0' }}>
            {result.Key_Entities.map((entity, i) => (
              <span
                key={i}
                style={{
                  fontFamily: 'var(--font-serif)',
                  fontSize: '1rem',
                  padding: '6px 16px',
                  border: '1px solid rgba(0,0,0,0.12)',
                  borderRadius: '99px',
                  color: 'var(--text)',
                  background: 'rgba(0,0,0,0.02)',
                }}
              >
                {entity}
              </span>
            ))}
          </div>
        </Card>
      </div>


      {/* Global stat card */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.45 }}
        style={{
          width: '100%',
          maxWidth: '960px',
          background: '#000',
          borderRadius: '16px',
          padding: '28px 36px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '20px',
          marginBottom: '40px',
        }}
      >
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1rem',
            flexShrink: 0,
            marginTop: '2px',
          }}
        >
          ✦
        </div>
        <div>
          <p
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.68rem',
              letterSpacing: '0.2em',
              color: 'rgba(255,255,255,0.4)',
              textTransform: 'uppercase',
              marginBottom: '8px',
            }}
          >
            Global Insight
          </p>
          <p
            style={{
              fontFamily: 'var(--font-serif)',
              fontSize: '1.1rem',
              color: '#fff',
              lineHeight: 1.6,
              fontStyle: 'italic',
            }}
          >
            "{result.Global_Stat}"
          </p>
        </div>
      </motion.div>

      {/* Reset */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        id="analyze-again-btn"
        onClick={onReset}
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.78rem',
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'var(--accent)',
          background: 'transparent',
          border: '1px solid rgba(111,111,111,0.3)',
          borderRadius: '99px',
          padding: '12px 36px',
          cursor: 'pointer',
          transition: 'all 0.25s',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = '#000';
          (e.currentTarget as HTMLElement).style.color = '#fff';
          (e.currentTarget as HTMLElement).style.borderColor = '#000';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = 'transparent';
          (e.currentTarget as HTMLElement).style.color = 'var(--accent)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(111,111,111,0.3)';
        }}
      >
        Analyze Another Dream
      </motion.button>
    </section>
  );
}

/* ── Glass Card wrapper ──────────────────────────────────── */
function Card({
  children,
  label,
  delay = 0,
}: {
  children: React.ReactNode;
  label: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, delay }}
      className="glass"
      style={{
        padding: '32px 28px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '20px',
      }}
    >
      <p
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.68rem',
          letterSpacing: '0.2em',
          color: 'var(--accent)',
          textTransform: 'uppercase',
          alignSelf: 'flex-start',
        }}
      >
        {label}
      </p>
      {children}
    </motion.div>
  );
}
