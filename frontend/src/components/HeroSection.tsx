import { useEffect, useRef } from 'react';

const VIDEO_URL =
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260328_083109_283f3553-e28f-428b-a723-d639c617eb2b.mp4';

const FADE_IN_DURATION = 0.5;  // seconds
const FADE_OUT_BEFORE = 0.5;  // seconds before end
const RESET_GAP_MS = 100;  // ms gap between loops

interface HeroSectionProps {
  onExplore: () => void;
  onGlobalInsights: () => void;
}

export default function HeroSection({ onExplore, onGlobalInsights }: HeroSectionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const loop = () => {
      const { currentTime, duration } = video;
      if (!duration) { rafRef.current = requestAnimationFrame(loop); return; }

      const timeLeft = duration - currentTime;

      // Fade-in
      if (currentTime < FADE_IN_DURATION) {
        video.style.opacity = String(currentTime / FADE_IN_DURATION);
      }
      // Fade-out
      else if (timeLeft < FADE_OUT_BEFORE) {
        video.style.opacity = String(Math.max(0, timeLeft / FADE_OUT_BEFORE));
      }
      // Fully visible
      else {
        video.style.opacity = '1';
      }

      // Loop trigger
      if (timeLeft < 0.08) {
        video.style.opacity = '0';
        video.pause();
        setTimeout(() => {
          video.currentTime = 0;
          video.play().catch(() => { });
        }, RESET_GAP_MS);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    video.play().catch(() => { });
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <section
      id="hero"
      style={{
        position: 'relative',
        width: '100%',
        height: '100svh',
        overflow: 'hidden',
        background: '#000',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* ── Background video ── */}
      <video
        ref={videoRef}
        src={VIDEO_URL}
        muted
        playsInline
        preload="auto"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          opacity: 0,
          transition: `opacity ${FADE_IN_DURATION}s ease`,
          zIndex: 0,
        }}
      />

      {/* ── Dark overlay ── */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.35) 100%)',
          zIndex: 1,
        }}
      />

      {/* ── Navigation ── */}
      <nav
        className="fade-rise"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '28px 48px',
          zIndex: 2,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-serif)',
            fontSize: '1.5rem',
            color: '#fff',
            letterSpacing: '-0.01em',
          }}
        >
          
        </span>
        <div style={{ display: 'flex', gap: '36px' }}>
          {['Global Insights'].map((item) => (
            <a
              key={item}
              href="#"
              style={{
                fontFamily: 'var(--font-sans)',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: 'rgba(255,255,255,0.75)',
                textDecoration: 'none',
                letterSpacing: '0.04em',
                transition: 'color 0.2s',
              }}
              onMouseEnter={(e) => ((e.target as HTMLElement).style.color = '#fff')}
              onMouseLeave={(e) => ((e.target as HTMLElement).style.color = 'rgba(255,255,255,0.75)')}
              onClick={(e) => { e.preventDefault(); onGlobalInsights(); }}
            >
              {item}
            </a>
          ))}
        </div>
      </nav>

      {/* ── Hero copy ── */}
      <div
        style={{
          position: 'relative',
          zIndex: 2,
          textAlign: 'center',
          padding: '0 24px',
          maxWidth: '720px',
        }}
      >
        <p
          className="fade-rise"
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.78rem',
            fontWeight: 700,
            letterSpacing: '0.2em',
            color: 'rgba(255,255,255,0.95)',
            textTransform: 'uppercase',
            marginBottom: '20px',
          }}
        >
          Dreamscape Mapper
        </p>

        <h1
          className="fade-rise-delay"
          style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 'clamp(2.4rem, 6vw, 4.5rem)',
            fontWeight: 400,
            color: '#fff',
            lineHeight: 1.15,
            letterSpacing: '-0.02em',
            marginBottom: '24px',
          }}
        >
          Beyond <em style={{ color: '#6F6F6F', fontStyle: 'italic' }}>silence</em>,
          {' '}we build the{' '}
          <em style={{ color: '#6F6F6F', fontStyle: 'italic' }}>eternal</em>.
        </h1>

        <p
          className="fade-rise-delay-2"
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '1rem',
            fontWeight: 500,
            color: 'rgba(255,255,255,0.85)',
            lineHeight: 1.7,
            marginBottom: '44px',
            maxWidth: '480px',
            margin: '0 auto 44px',
          }}
        >
          An NLP-powered engine that decodes the emotional architecture
          of your dreams, one symbol at a time.
        </p>

        <button
          className="fade-rise-delay-2"
          onClick={onExplore}
          style={{
            fontFamily: 'var(--font-sans)',
            fontSize: '0.85rem',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: '#000',
            background: '#fff',
            border: 'none',
            borderRadius: '99px',
            padding: '14px 40px',
            cursor: 'pointer',
            transition: 'transform 0.2s, background 0.2s',
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.transform = 'scale(1.04)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = 'scale(1)'; }}
        >
          Enter the Chamber
        </button>
      </div>

      {/* ── Scroll indicator ── */}
      <div
        style={{
          position: 'absolute',
          bottom: '32px',
          zIndex: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <div
          style={{
            width: '1px',
            height: '48px',
            background: 'linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0.5))',
            animation: 'fade-rise 2s infinite alternate ease-in-out',
          }}
        />
        <span style={{ fontFamily: 'var(--font-sans)', fontSize: '0.65rem', color: 'rgba(255,255,255,0.4)', letterSpacing: '0.15em' }}>
          SCROLL
        </span>
      </div>
    </section>
  );
}
