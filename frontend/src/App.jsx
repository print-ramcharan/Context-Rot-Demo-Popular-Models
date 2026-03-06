import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Layout/Header';
import FileDropzone from './components/Upload/FileDropzone';
import QueryBar from './components/Chat/QueryBar';
import ComparisonView from './components/Comparison/ComparisonView';
import { AlertCircle } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

/* ── Step label ── */
function StepLabel({ number, label, active }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '11px',
        fontWeight: 500,
        color: active ? 'var(--purple-400)' : 'var(--text-muted)',
        letterSpacing: '0.08em',
        transition: 'color 0.3s ease',
        userSelect: 'none',
      }}>
        0{number}
      </span>
      <div style={{
        height: '1px',
        width: '24px',
        background: active ? 'var(--purple-500)' : 'var(--border)',
        transition: 'background 0.3s ease',
      }} />
      <span style={{
        fontSize: '12px',
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        color: active ? 'var(--text-secondary)' : 'var(--text-muted)',
        transition: 'color 0.3s ease',
      }}>
        {label}
      </span>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [loadingStandard, setLoadingStandard] = useState(false);
  const [loadingRag, setLoadingRag] = useState(false);
  const [error, setError] = useState('');
  const [responses, setResponses] = useState({ standard: null, rag: null });
  const [sources, setSources] = useState([]);
  const [documentUploaded, setDocumentUploaded] = useState(false);

  const handleFileUpload = (file) => {
    setDocumentUploaded(true);
    setError('');
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) { setError('Please enter a question'); return; }

    setLoadingStandard(true);
    setLoadingRag(true);
    setError('');
    setResponses({ standard: null, rag: null });
    setSources([]);

    // Independent fetch for Standard path
    axios.post(`${API_BASE}/query/standard`, { user_query: query })
      .then(({ data }) => {
        if (data.status !== 'success') throw new Error(data.detail || 'Standard query failed');
        setResponses(prev => ({ ...prev, standard: data.response }));
      })
      .catch(err => {
        const message = err?.response?.data?.detail || err.message || 'Error occurred';
        setResponses(prev => ({ ...prev, standard: { text: `Error: ${message}` } }));
      })
      .finally(() => setLoadingStandard(false));

    // Independent fetch for RAG path
    axios.post(`${API_BASE}/query/rag`, { user_query: query })
      .then(({ data }) => {
        if (data.status !== 'success') throw new Error(data.detail || 'RAG query failed');
        setResponses(prev => ({ ...prev, rag: data.response }));
        if (data.sources) setSources(data.sources);
      })
      .catch(err => {
        const message = err?.response?.data?.detail || err.message || 'Error occurred';
        setResponses(prev => ({ ...prev, rag: { text: `Error: ${message}` } }));
      })
      .finally(() => setLoadingRag(false));
  };

  const hasResults = responses.standard || responses.rag || loadingStandard || loadingRag;

  return (
    <div className="bg-radial-glow" style={{ minHeight: '100vh' }}>
      <Header />

      <main style={{
        maxWidth: '1100px',
        margin: '0 auto',
        padding: '48px 24px 80px',
        display: 'flex',
        flexDirection: 'column',
        gap: '48px',
      }}>

        {/* ── Hero intro ── */}
        <div className="animate-fade-up" style={{ textAlign: 'center', maxWidth: '620px', margin: '0 auto' }}>
          <h2 style={{
            fontSize: 'clamp(26px, 4vw, 38px)',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.2,
            marginBottom: '14px',
          }}>
            <span className="gradient-text">Context Rot</span>
            {' '}vs RAG
          </h2>
          <p style={{
            fontSize: '15px',
            color: 'var(--text-secondary)',
            lineHeight: '1.7',
          }}>
            Upload a document, ask a question, and observe how context degradation
            impacts standard LLM responses — compared against retrieval-augmented generation.
          </p>
        </div>

        {/* ── Step 1: Upload ── */}
        <section className="animate-fade-up" style={{ animationDelay: '80ms' }}>
          <StepLabel number={1} label="Upload Document" active={true} />
          <FileDropzone onUpload={handleFileUpload} />
        </section>

        {/* ── Step 2: Query ── */}
        <section className="animate-fade-up" style={{ animationDelay: '160ms' }}>
          <StepLabel number={2} label="Ask a Question" active={documentUploaded} />
          <QueryBar
            value={query}
            onChange={setQuery}
            onSubmit={handleQuery}
            loading={loadingStandard || loadingRag}
          />

          {error && (
            <div className="animate-fade-in" style={{
              marginTop: '12px',
              display: 'flex', alignItems: 'flex-start', gap: '10px',
              padding: '14px 16px',
              borderRadius: '10px',
              border: '1px solid rgba(239,68,68,0.3)',
              background: 'rgba(239,68,68,0.06)',
            }}>
              <AlertCircle size={16} style={{ color: 'var(--red-400)', flexShrink: 0, marginTop: '1px' }} />
              <div>
                <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--red-400)', marginBottom: '2px' }}>Error</p>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* ── Step 3: Results ── */}
        <section className="animate-fade-up" style={{ animationDelay: '240ms' }}>
          <StepLabel number={3} label="Compare Responses" active={!!hasResults} />

          {hasResults ? (
            <ComparisonView
              standard={responses.standard}
              rag={responses.rag}
              loadingStandard={loadingStandard}
              loadingRag={loadingRag}
              sources={sources}
            />
          ) : (
            <div style={{
              borderRadius: '14px',
              border: '2px dashed var(--border)',
              padding: '56px 24px',
              textAlign: 'center',
              background: 'var(--bg-surface)',
            }}>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Results will appear here after you upload a document and submit a query.
              </p>
            </div>
          )}
        </section>
      </main>

      {/* ── Footer ── */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '20px 24px',
        textAlign: 'center',
      }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Built with{' '}
          {['React', 'FastAPI', 'Gemini', 'FAISS'].map((t, i) => (
            <span key={t}>
              {i > 0 && <span style={{ margin: '0 6px', opacity: 0.4 }}>·</span>}
              <span style={{
                padding: '2px 7px', borderRadius: '5px',
                border: '1px solid var(--border)',
                background: 'var(--bg-surface)',
                fontSize: '11px',
                fontFamily: "'JetBrains Mono', monospace",
              }}>{t}</span>
            </span>
          ))}
        </p>
      </footer>
    </div>
  );
}