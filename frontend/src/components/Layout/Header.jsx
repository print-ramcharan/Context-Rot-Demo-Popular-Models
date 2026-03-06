import React, { useEffect, useState } from 'react';
import { Layers3, Moon, Sun, Cpu } from 'lucide-react';

export default function Header() {
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') || 'dark';
    }
    return 'dark';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('light');
    } else {
      root.classList.remove('light');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        width: '100%',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(7, 5, 15, 0.85)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
      }}
      className="light-header"
    >
      <style>{`
        :root.light header.light-header {
          background: rgba(248, 245, 255, 0.90) !important;
        }
      `}</style>

      <div style={{
        maxWidth: '1280px',
        margin: '0 auto',
        padding: '0 24px',
        height: '64px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>

        {/* Logo + Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px', height: '36px',
            borderRadius: '10px',
            border: '1px solid var(--border-strong)',
            background: 'linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(139,92,246,0.05) 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Layers3 size={18} style={{ color: 'var(--purple-400)' }} />
          </div>
          <div>
            <h1 style={{
              fontSize: '15px',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              lineHeight: 1.2,
            }}>
              Context Rot Lab
            </h1>
            <p style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              letterSpacing: '0.04em',
              marginTop: '1px',
            }}>
              RAG vs Standard · 2026
            </p>
          </div>
        </div>

        {/* Right side */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Stack badges */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            fontSize: '11px', color: 'var(--text-muted)',
            fontFamily: "'JetBrains Mono', monospace",
          }}
            className="stack-badges"
          >
            <style>{`
              @media (max-width: 600px) { .stack-badges { display: none !important; } }
            `}</style>
            {['Gemini', 'FAISS', 'MiniLM'].map((label, i) => (
              <span key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                {i > 0 && <span style={{ opacity: 0.3 }}>·</span>}
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: '4px',
                  padding: '3px 8px',
                  borderRadius: '6px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-surface)',
                  color: 'var(--text-secondary)',
                }}>
                  <span className="live-dot" style={{
                    background: i === 0 ? 'var(--purple-400)' : i === 1 ? 'var(--emerald-400)' : 'var(--orange-400)',
                  }} />
                  {label}
                </span>
              </span>
            ))}
          </div>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            style={{
              width: '36px', height: '36px',
              borderRadius: '8px',
              border: '1px solid var(--border)',
              background: 'var(--bg-surface)',
              color: 'var(--text-secondary)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--border-strong)';
              e.currentTarget.style.color = 'var(--purple-400)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
          >
            {theme === 'dark'
              ? <Sun size={16} />
              : <Moon size={16} />
            }
          </button>
        </div>
      </div>
    </header>
  );
}