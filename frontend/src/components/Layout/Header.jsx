import React, { useEffect, useState } from 'react';
import { Layers3, Moon, Sun, Github, ExternalLink } from 'lucide-react';

export default function Header({ page = 'home' }) {
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

  const toggleTheme = () => setTheme(current => current === 'dark' ? 'light' : 'dark');

  return (
    <header className="site-header">
      <div className="header-inner">
        <a className="brand-link" href="#/home" style={{ display: 'flex', alignItems: 'center', gap: '12px', textDecoration: 'none', color: 'inherit' }}>
          <div className="brand-mark" style={{ width: '40px', height: '40px', borderRadius: '12px', background: 'var(--purple-600)', display: 'grid', placeItems: 'center', color: 'white' }}>
            <Layers3 size={20} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '18px', fontWeight: '800', lineHeight: '1' }}>Context Rot Lab</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: '600' }}>PRECISION RAG ENGINE</span>
          </div>
        </a>

        <nav style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
          <a 
            href="#/home" 
            style={{ 
              textDecoration: 'none', 
              color: page === 'home' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: '600',
              fontSize: '14px',
              transition: 'color 0.2s'
            }}
          >
            Home
          </a>
          <a 
            href="#/app" 
            style={{ 
              textDecoration: 'none', 
              color: page === 'app' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontWeight: '600',
              fontSize: '14px',
              transition: 'color 0.2s'
            }}
          >
            Workspace
          </a>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-secondary)' }}>
            <Github size={20} />
          </a>
          <button
            onClick={toggleTheme}
            style={{ 
              background: 'none', 
              border: 'none', 
              color: 'var(--text-secondary)', 
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          {page !== 'app' && (
            <a href="#/app" className="btn-primary" style={{ padding: '8px 20px', fontSize: '13px' }}>
              Get Started
            </a>
          )}
        </div>
      </div>
    </header>
  );
}
