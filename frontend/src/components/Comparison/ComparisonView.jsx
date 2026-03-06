import React, { useState, useEffect } from 'react';
import { AlertTriangle, Zap, BookOpen, Clock, Cpu, ChevronDown, TrendingDown, DollarSign, Activity } from 'lucide-react';

/* ── Skeleton row ── */
function SkeletonRow({ width = '100%', delay = 0 }) {
  return (
    <div className="skeleton" style={{
      height: '14px',
      width,
      marginBottom: '10px',
      animationDelay: `${delay}ms`,
    }} />
  );
}

/* ── Streaming text that reveals word-by-word ── */
function StreamingText({ text }) {
  const [displayed, setDisplayed] = useState('');

  useEffect(() => {
    if (!text) return;
    setDisplayed('');
    const words = text.split(' ');
    let idx = 0;
    const interval = setInterval(() => {
      idx++;
      setDisplayed(words.slice(0, idx).join(' '));
      if (idx >= words.length) clearInterval(interval);
    }, 18);
    return () => clearInterval(interval);
  }, [text]);

  return (
    <p style={{
      fontSize: '13.5px',
      lineHeight: '1.75',
      color: 'var(--text-secondary)',
      whiteSpace: 'pre-wrap',
      fontFamily: 'inherit',
    }}>
      {displayed}
      {displayed.length < text.length && (
        <span style={{
          display: 'inline-block',
          width: '2px', height: '14px',
          background: 'var(--purple-400)',
          marginLeft: '2px',
          verticalAlign: 'text-bottom',
          animation: 'pulseDot 0.8s ease-in-out infinite',
        }} />
      )}
    </p>
  );
}

/* ── Meta chip ── */
function MetaChip({ icon: Icon, label }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '5px',
      padding: '3px 8px',
      borderRadius: '6px',
      border: '1px solid var(--border)',
      background: 'var(--bg-base)',
      fontSize: '11px',
      color: 'var(--text-muted)',
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <Icon size={11} />
      {label}
    </span>
  );
}

/* ── Single response panel ── */
function ResponsePane({ type, title, subtitle, icon: Icon, response, loading, sources = [], showSources }) {
  const isStandard = type === 'standard';
  const [contextOpen, setContextOpen] = useState(false);

  const accentColor = isStandard ? 'var(--red-400)' : 'var(--emerald-400)';
  const accentBg = isStandard ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)';
  const accentBorder = isStandard ? 'rgba(239,68,68,0.22)' : 'rgba(16,185,129,0.22)';
  const badgeLabel = isStandard ? 'Degraded' : 'Enhanced';

  return (
    <div className="animate-fade-up" style={{
      flex: 1,
      minWidth: 0,
      borderRadius: '14px',
      border: `1px solid ${accentBorder}`,
      background: 'var(--bg-surface)',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      boxShadow: `0 2px 12px ${isStandard ? 'rgba(239,68,68,0.06)' : 'rgba(16,185,129,0.06)'}`,
    }}>

      {/* Panel header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: `1px solid ${accentBorder}`,
        background: accentBg,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px',
            borderRadius: '8px',
            border: `1px solid ${accentBorder}`,
            background: isStandard ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={15} style={{ color: accentColor }} />
          </div>
          <div>
            <h3 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.2 }}>
              {title}
            </h3>
            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '1px' }}>{subtitle}</p>
          </div>
        </div>

        {!loading && response && (
          <span style={{
            padding: '3px 10px',
            borderRadius: '20px',
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: accentColor,
            border: `1px solid ${accentBorder}`,
            background: accentBg,
          }}>
            {badgeLabel}
          </span>
        )}
      </div>

      {/* Body */}
      <div style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>

        {/* Response text box */}
        <div style={{
          minHeight: '180px',
          borderRadius: '10px',
          border: '1px solid var(--border)',
          background: 'var(--bg-card)',
          padding: '16px',
          flex: 1,
        }}>
          {loading ? (
            <>
              <SkeletonRow width="72%" delay={0} />
              <SkeletonRow width="100%" delay={80} />
              <SkeletonRow width="88%" delay={160} />
              <SkeletonRow width="95%" delay={240} />
              <SkeletonRow width="60%" delay={320} />
            </>
          ) : response?.text ? (
            <StreamingText text={response.text} />
          ) : (
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No response yet. Upload a document and ask a question.
            </p>
          )}
        </div>

        {/* Retrieved context (RAG only) */}
        {!isStandard && response?.context_used && (
          <div style={{
            borderRadius: '10px',
            border: '1px solid rgba(16,185,129,0.2)',
            overflow: 'hidden',
          }}>
            <button
              onClick={() => setContextOpen(o => !o)}
              style={{
                width: '100%',
                padding: '10px 14px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: 'rgba(16,185,129,0.06)',
                border: 'none',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: 'var(--emerald-400)' }}>
                <BookOpen size={13} />
                Retrieved Context
              </span>
              <ChevronDown size={13} style={{
                color: 'var(--text-muted)',
                transform: contextOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
              }} />
            </button>
            {contextOpen && (
              <div style={{
                padding: '12px 14px',
                borderTop: '1px solid rgba(16,185,129,0.15)',
                background: 'var(--bg-card)',
              }}>
                <pre style={{
                  fontSize: '11.5px',
                  color: 'var(--text-secondary)',
                  whiteSpace: 'pre-wrap',
                  fontFamily: "'JetBrains Mono', monospace",
                  lineHeight: '1.7',
                }}>
                  {response.context_used}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Sources (RAG only) */}
        {showSources && sources.length > 0 && (
          <div>
            <p style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Source Citations
            </p>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {sources.map((source, i) => (
                <li key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-card)',
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                  lineHeight: '1.6',
                }}>
                  <span style={{
                    flexShrink: 0,
                    width: '18px', height: '18px',
                    borderRadius: '4px',
                    background: 'rgba(16,185,129,0.12)',
                    color: 'var(--emerald-400)',
                    fontSize: '10px', fontWeight: 700,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontFamily: "'JetBrains Mono', monospace",
                  }}>
                    {i + 1}
                  </span>
                  {source}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metadata row */}
        {response && (
          <div style={{
            display: 'flex', flexWrap: 'wrap', gap: '6px',
            paddingTop: '12px',
            borderTop: '1px solid var(--border)',
          }}>
            <MetaChip icon={Cpu} label={response.model || 'gemini'} />
            <MetaChip icon={Clock} label={`${response.latency_ms?.toFixed?.(0) || 0} ms`} />
            <MetaChip icon={() => <span style={{ fontSize: '10px' }}>T</span>}
              label={`${response.tokens_used?.total || 0} tokens`} />
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Performance Metrics Banner ── */
function PerformanceMetrics({ standard, rag }) {
  if (!standard || !rag) return null;

  const stdTokens = standard.tokens_used?.total || 0;
  const ragTokens = rag.tokens_used?.total || 0;

  const stdTime = standard.latency_ms || 0;
  const ragTime = rag.latency_ms || 0;

  // Gemini 1.5/2.5 Flash approx pricing: $0.075 per 1M input tokens
  const COST_PER_1M_TOKENS = 0.075;
  const stdCost = (stdTokens / 1000000) * COST_PER_1M_TOKENS;
  const ragCost = (ragTokens / 1000000) * COST_PER_1M_TOKENS;

  // Savings math
  const tokenSavings = stdTokens > 0 ? ((stdTokens - ragTokens) / stdTokens * 100).toFixed(1) : 0;
  const timeSavings = stdTime > 0 ? ((stdTime - ragTime) / stdTime * 100).toFixed(1) : 0;
  const costSavingsMult = ragCost > 0 ? Math.round(stdCost / ragCost) : 0;

  return (
    <div className="animate-fade-in" style={{
      marginBottom: '24px',
      padding: '20px 24px',
      borderRadius: '16px',
      background: 'linear-gradient(145deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)',
      border: '1px solid rgba(16,185,129,0.2)',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Activity size={18} style={{ color: 'var(--emerald-500)' }} />
        <h3 style={{ fontSize: '15px', fontWeight: 700, margin: 0, color: 'var(--emerald-400)' }}>
          RAG Efficiency Gains
        </h3>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        <div style={{ background: 'var(--bg-surface)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: 'var(--text-muted)' }}>
            <Clock size={14} /> <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Time Saved</span>
          </div>
          <p style={{ fontSize: '24px', fontWeight: 800, margin: 0, color: 'var(--text-primary)' }}>
            {timeSavings}% <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>faster</span>
          </p>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', fontFamily: "'JetBrains Mono', monospace" }}>
            {(stdTime / 1000).toFixed(1)}s &rarr; {(ragTime / 1000).toFixed(1)}s
          </p>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: 'var(--text-muted)' }}>
            <TrendingDown size={14} /> <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Compute Saved</span>
          </div>
          <p style={{ fontSize: '24px', fontWeight: 800, margin: 0, color: 'var(--text-primary)' }}>
            {tokenSavings}% <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>fewer tokens</span>
          </p>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', fontFamily: "'JetBrains Mono', monospace" }}>
            {stdTokens.toLocaleString()} &rarr; {ragTokens.toLocaleString()}
          </p>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: 'var(--text-muted)' }}>
            <DollarSign size={14} /> <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Cost Reduction</span>
          </div>
          <p style={{ fontSize: '24px', fontWeight: 800, margin: 0, color: 'var(--text-primary)' }}>
            {costSavingsMult}x <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>cheaper</span>
          </p>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', fontFamily: "'JetBrains Mono', monospace" }}>
            ${stdCost.toFixed(5)} &rarr; ${ragCost.toFixed(6)}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ── Public export ── */
export default function ComparisonView({ standard, rag, loadingStandard, loadingRag, sources = [] }) {
  const isComplete = standard && rag && !loadingStandard && !loadingRag;

  return (
    <div>
      {isComplete && <PerformanceMetrics standard={standard} rag={rag} />}
      <div style={{
        display: 'flex',
        gap: '16px',
        alignItems: 'stretch',
      }}
        className="comparison-grid"
      >
        <style>{`
        @media (max-width: 900px) {
          .comparison-grid { flex-direction: column !important; }
        }
      `}</style>

        <ResponsePane
          type="standard"
          title="Context Rot"
          subtitle="No retrieved context"
          icon={AlertTriangle}
          response={standard}
          loading={loadingStandard}
        />
        <ResponsePane
          type="rag"
          title="RAG Optimized"
          subtitle="With retrieved context"
          icon={Zap}
          response={rag}
          loading={loadingRag}
          sources={sources}
          showSources={true}
        />
      </div>
    </div>
  );
}