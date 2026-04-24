import { motion } from 'framer-motion';
import { useState, useRef } from 'react';

interface DreamChamberProps {
  onSubmit: (dreamText: string) => void;
  isLoading: boolean;
}

export default function DreamChamber({ onSubmit, isLoading }: DreamChamberProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    if (text.trim() && !isLoading) {
      onSubmit(text.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <section
      id="chamber"
      style={{
        minHeight: '100svh',
        background: 'var(--bg)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '80px 24px',
      }}
    >
      {/* Label */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.72rem',
          letterSpacing: '0.22em',
          color: 'var(--accent)',
          textTransform: 'uppercase',
          marginBottom: '16px',
        }}
      >
        The Chamber
      </motion.p>

      {/* Heading */}
      <motion.h2
        initial={{ opacity: 0, y: 28 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7, delay: 0.1 }}
        style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 'clamp(2rem, 4vw, 3.2rem)',
          fontWeight: 400,
          color: 'var(--text)',
          letterSpacing: '-0.02em',
          textAlign: 'center',
          lineHeight: 1.2,
          marginBottom: '12px',
          maxWidth: '560px',
        }}
      >
        Describe your dream.
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
        style={{
          fontFamily: 'var(--font-sans)',
          fontSize: '0.9rem',
          color: 'var(--accent)',
          textAlign: 'center',
          marginBottom: '48px',
        }}
      >
        Write freely. The pipeline will map its hidden architecture.
      </motion.p>

      {/* Main input box */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay: 0.25, ease: [0.4, 0, 0.2, 1] }}
        style={{ width: '100%', maxWidth: '700px' }}
      >
        <div
          style={{
            background: '#fff',
            border: '1.5px solid',
            borderColor: text.length > 0 ? '#000' : 'rgba(111,111,111,0.25)',
            borderRadius: '20px',
            overflow: 'hidden',
            boxShadow: text.length > 0
              ? '0 8px 40px rgba(0,0,0,0.08)'
              : '0 2px 12px rgba(0,0,0,0.04)',
            transition: 'border-color 0.3s, box-shadow 0.3s',
          }}
        >
          {isLoading ? (
            /* Shimmer loading state */
            <div style={{ padding: '32px 28px' }}>
              {[1, 0.7, 0.9, 0.5].map((w, i) => (
                <div
                  key={i}
                  className="shimmer"
                  style={{
                    height: '18px',
                    width: `${w * 100}%`,
                    marginBottom: i < 3 ? '14px' : 0,
                    borderRadius: '6px',
                  }}
                />
              ))}
              <p
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.8rem',
                  color: 'var(--accent)',
                  marginTop: '24px',
                  textAlign: 'center',
                  letterSpacing: '0.06em',
                }}
              >
                Mapping your dreamscape…
              </p>
            </div>
          ) : (
            <textarea
              ref={textareaRef}
              id="dream-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={8}
              placeholder="Last night I was standing in a dark hallway when suddenly the mirror shattered and I started to fall…"
              style={{
                width: '100%',
                padding: '32px 28px',
                fontFamily: 'var(--font-sans)',
                fontSize: '1rem',
                lineHeight: 1.75,
                color: 'var(--text)',
                background: 'transparent',
                border: 'none',
                resize: 'none',
                outline: 'none',
              }}
            />
          )}

          {!isLoading && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 20px 16px',
                borderTop: '1px solid rgba(111,111,111,0.1)',
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.72rem',
                  color: 'var(--accent)',
                  letterSpacing: '0.04em',
                }}
              >
                {text.length > 0 ? `${text.split(/\s+/).filter(Boolean).length} words` : '⌘ + Enter to submit'}
              </span>
              <button
                id="analyze-btn"
                onClick={handleSubmit}
                disabled={!text.trim()}
                style={{
                  fontFamily: 'var(--font-sans)',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  letterSpacing: '0.08em',
                  color: text.trim() ? '#fff' : 'var(--accent)',
                  background: text.trim() ? '#000' : 'rgba(111,111,111,0.08)',
                  border: 'none',
                  borderRadius: '99px',
                  padding: '10px 28px',
                  cursor: text.trim() ? 'pointer' : 'not-allowed',
                  transition: 'all 0.25s',
                  textTransform: 'uppercase',
                }}
                onMouseEnter={(e) => {
                  if (text.trim()) (e.currentTarget as HTMLElement).style.transform = 'scale(1.04)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.transform = 'scale(1)';
                }}
              >
                Analyze
              </button>
            </div>
          )}
        </div>

        {!isLoading && (
          <p
            style={{
              fontFamily: 'var(--font-sans)',
              fontSize: '0.72rem',
              color: 'rgba(111,111,111,0.5)',
              textAlign: 'center',
              marginTop: '16px',
              letterSpacing: '0.04em',
            }}
          >
            Your dream is processed locally. Nothing is stored.
          </p>
        )}
      </motion.div>
    </section>
  );
}
