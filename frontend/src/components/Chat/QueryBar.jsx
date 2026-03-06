import React, { useRef } from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';

export default function QueryBar({ value, onChange, onSubmit, loading }) {
  const inputRef = useRef(null);

  return (
    <form
      onSubmit={onSubmit}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0',
        width: '100%',
        borderRadius: '12px',
        border: '1px solid var(--border-strong)',
        background: 'var(--bg-surface)',
        boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
        overflow: 'hidden',
        transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
      }}
      onFocusCapture={e => e.currentTarget.style.boxShadow = '0 0 0 3px rgba(139,92,246,0.18), 0 1px 3px rgba(0,0,0,0.12)'}
      onBlurCapture={e => e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.12)'}
    >
      {/* Left icon */}
      <div style={{
        padding: '0 14px',
        display: 'flex', alignItems: 'center',
        flexShrink: 0,
      }}>
        {loading
          ? (
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          )
          : <Sparkles size={16} style={{ color: 'var(--purple-400)' }} />
        }
      </div>

      {/* Input */}
      <input
        ref={inputRef}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Ask a question about your document…"
        disabled={loading}
        style={{
          flex: 1,
          padding: '14px 4px',
          fontSize: '14px',
          color: 'var(--text-primary)',
          background: 'transparent',
          border: 'none',
          outline: 'none',
          fontFamily: 'inherit',
        }}
      />

      {/* Submit button */}
      <button
        type="submit"
        disabled={loading || !value.trim()}
        style={{
          flexShrink: 0,
          margin: '6px',
          padding: '8px 18px',
          borderRadius: '8px',
          border: 'none',
          background: loading || !value.trim()
            ? 'var(--bg-hover)'
            : 'linear-gradient(135deg, var(--purple-600), var(--purple-500))',
          color: loading || !value.trim() ? 'var(--text-muted)' : '#fff',
          fontSize: '13px',
          fontWeight: 600,
          fontFamily: 'inherit',
          cursor: loading || !value.trim() ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', gap: '6px',
          transition: 'opacity 0.2s ease, background 0.2s ease',
          letterSpacing: '0.01em',
        }}
        onMouseEnter={e => {
          if (!loading && value.trim()) e.currentTarget.style.opacity = '0.9';
        }}
        onMouseLeave={e => { e.currentTarget.style.opacity = '1'; }}
      >
        {loading ? 'Analyzing…' : <>Query <ArrowRight size={13} /></>}
      </button>
    </form>
  );
}