import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Header from './components/Layout/Header';
import FileDropzone from './components/Upload/FileDropzone';
import QueryBar from './components/Chat/QueryBar';
import ComparisonView from './components/Comparison/ComparisonView';
import {
  ArrowRight,
  ArrowUpRight,
  Bot,
  Code2,
  Cpu,
  Database,
  Download,
  Github,
  Globe,
  Layers,
  Layout,
  MessageSquare,
  Zap,
  Twitter,
  Linkedin,
  Mail,
  Heart
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

const technologies = [
  { name: 'Gemini 1.5 Pro', desc: 'State-of-the-art multimodal reasoning', icon: Bot },
  { name: 'FAISS', desc: 'High-performance vector similarity search', icon: Database },
  { name: 'React 18', desc: 'Declarative component-based UI', icon: Layout },
  { name: 'FastAPI', desc: 'Python-based high-speed backend framework', icon: Zap },
  { name: 'Vite', desc: 'Modern frontend build tool for speed', icon: Cpu },
  { name: 'MiniLM-L6', desc: 'Efficient sentence embeddings', icon: Code2 },
];

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <h2 className="font-heading">Context Rot Lab</h2>
          <p>
            Bridging the gap between standard LLM responses and precision RAG architectures.
          </p>
          <div className="social-links" style={{ marginTop: '24px' }}>
            <a href="#" className="social-link"><Twitter size={20} /></a>
            <a href="#" className="social-link"><Github size={20} /></a>
            <a href="#" className="social-link"><Linkedin size={20} /></a>
          </div>
        </div>
        <div className="footer-nav">
          <h4 className="font-heading">Product</h4>
          <ul className="footer-links">
            <li><a href="#/home" className="footer-link">Home</a></li>
            <li><a href="#/app" className="footer-link">Workspace</a></li>
            <li><a href="#" className="footer-link">Extension</a></li>
            <li><a href="#" className="footer-link">Documentation</a></li>
          </ul>
        </div>
        <div className="footer-nav">
          <h4 className="font-heading">Company</h4>
          <ul className="footer-links">
            <li><a href="#" className="footer-link">About Us</a></li>
            <li><a href="#" className="footer-link">Blog</a></li>
            <li><a href="#" className="footer-link">Partners</a></li>
            <li><a href="#" className="footer-link">Contact</a></li>
          </ul>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; 2026 Context Rot Lab. All rights reserved.</p>
      </div>
    </footer>
  );
}

function HomePage({ onEnterApp }) {
  return (
    <div className="animate-reveal">
      <section className="hero-section">
        <div className="hero-tag animate-scale">NEW: BROWSER MEMORY EXTENSION v1.0</div>
        <h1 className="hero-title animate-reveal stagger-1">
          Stop Context Rot.<br />Enhance Your RAG.
        </h1>
        <p className="hero-description animate-reveal stagger-2">
          A dual-path evaluation studio designed to compare standard LLM responses
          against high-precision RAG architectures in real-time.
        </p>
        <div className="hero-actions animate-reveal stagger-3">
          <button className="btn-primary" onClick={onEnterApp}>
            Try Workspace <ArrowRight size={18} />
          </button>
          <button className="btn-secondary">
            <Download size={18} /> Download Extension
          </button>
        </div>
      </section>

      <section className="tech-section">
        <div className="section-label animate-reveal stagger-4">Powered by modern stack</div>
        <div className="tech-grid">
          {technologies.map((tech, i) => (
            <div key={tech.name} className={`tech-item animate-reveal stagger-${(i % 4) + 1}`}>
              <tech.icon className="tech-icon" />
              <h3 className="font-heading">{tech.name}</h3>
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)', textAlign: 'center' }}>{tech.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-section animate-reveal" style={{ padding: '80px 0', textAlign: 'center' }}>
        <div className="glass-container" style={{ padding: '60px', overflow: 'hidden', position: 'relative' }}>
          <h2 className="hero-title" style={{ fontSize: '48px', marginBottom: '16px' }}>Ready to optimize your context?</h2>
          <p className="hero-description" style={{ marginBottom: '32px' }}>Join the future of retrieval-augmented generation.</p>
          <button className="btn-primary" onClick={onEnterApp} style={{ padding: '16px 40px', fontSize: '18px' }}>
            Open the Workspace <Zap size={20} />
          </button>
        </div>
      </section>

      <Footer />
    </div>
  );
}

function WorkspacePage({
  query,
  setQuery,
  loadingStandard,
  loadingRag,
  error,
  responses,
  sources,
  documentUploaded,
  onFileUpload,
  onQuery,
}) {
  const hasResults = responses.standard || responses.rag || loadingStandard || loadingRag;

  return (
    <div className="workspace-container animate-reveal">
      <div className="workspace-header">
        <div>
          <h1 className="workspace-title font-heading">Workspace</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Upload your source and query the precision engine.</p>
        </div>
        <div className="workspace-stats">
          <div className="stat-item">
            <div className="stat-value">2.4s</div>
            <div className="stat-label">Avg Latency</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">98%</div>
            <div className="stat-label">Accuracy</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
        <div className="premium-card">
          <h3 className="font-heading" style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} /> 1. Upload Source
          </h3>
          <FileDropzone onUpload={onFileUpload} />
        </div>

        <div className="premium-card" style={{ opacity: documentUploaded ? 1 : 0.5, transition: 'opacity 0.3s' }}>
          <h3 className="font-heading" style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <MessageSquare size={18} /> 2. Expert Query
          </h3>
          <QueryBar
            value={query}
            onChange={setQuery}
            onSubmit={onQuery}
            loading={loadingStandard || loadingRag}
            disabled={!documentUploaded}
          />
          {error && <p style={{ color: '#ef4444', fontSize: '12px', marginTop: '8px' }}>{error}</p>}
        </div>

        <div className="premium-card" style={{ minHeight: '500px' }}>
          <h3 className="font-heading" style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Zap size={18} /> 3. Comparative Results
          </h3>
          {hasResults ? (
            <ComparisonView
              standard={responses.standard}
              rag={responses.rag}
              loadingStandard={loadingStandard}
              loadingRag={loadingRag}
              sources={sources}
            />
          ) : (
            <div style={{ height: '300px', display: 'grid', placeItems: 'center', color: 'var(--text-muted)' }}>
              <div style={{ textAlign: 'center' }}>
                <Globe size={48} style={{ marginBottom: '16px', opacity: 0.2 }} />
                <p>Upload a document and ask a question to see the comparative beauty.</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: '80px' }}>
        <Footer />
      </div>
    </div>
  );
}

function getRouteFromHash() {
  if (typeof window === 'undefined') return 'home';
  const rawHash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
  return rawHash === 'app' ? 'app' : 'home';
}

export default function App() {
  const [route, setRoute] = useState(getRouteFromHash());
  const [query, setQuery] = useState('');
  const [loadingStandard, setLoadingStandard] = useState(false);
  const [loadingRag, setLoadingRag] = useState(false);
  const [error, setError] = useState('');
  const [responses, setResponses] = useState({ standard: null, rag: null });
  const [sources, setSources] = useState([]);
  const [documentUploaded, setDocumentUploaded] = useState(false);

  useEffect(() => {
    const syncRoute = () => {
      setRoute(getRouteFromHash());
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    if (!window.location.hash) {
      window.location.hash = '#/home';
    }

    syncRoute();
    window.addEventListener('hashchange', syncRoute);
    return () => window.removeEventListener('hashchange', syncRoute);
  }, []);

  const navigate = (nextRoute) => {
    window.location.hash = `#/${nextRoute}`;
  };

  const handleFileUpload = () => {
    setDocumentUploaded(true);
    setError('');
  };

  const streamQuery = async (endpoint, type) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_query: query })
      });

      if (!response.ok) throw new Error(`${type} query failed`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            
            if (data.type === 'metadata') {
              setResponses(prev => ({
                ...prev,
                [type]: { 
                  ...prev[type], 
                  model: data.model,
                  context_used: data.context || data.context_used,
                  text: (prev[type]?.text || '')
                }
              }));
              if (type === 'rag' && data.sources) setSources(data.sources);
            } 
            else if (data.type === 'text') {
              setResponses(prev => ({
                ...prev,
                [type]: { 
                  ...prev[type], 
                  text: (prev[type]?.text || '') + (data.text || '')
                }
              }));
            } 
            else if (data.type === 'final') {
              setResponses(prev => ({
                ...prev,
                [type]: { 
                  ...prev[type], 
                  tokens_used: data.tokens,
                  latency_ms: data.latency_ms
                }
              }));
            }
            else if (data.type === 'error') {
              throw new Error(data.detail);
            }
          } catch (e) {
            console.error("Error parsing stream line:", e);
            if (e.message && e.message !== 'Unexpected end of JSON input') {
               throw e; // Re-throw to be caught by outer catch if it's a logic error
            }
          }
        }
      }
    } catch (err) {
      const message = err.message || 'Error occurred';
      setResponses(prev => ({ 
        ...prev, 
        [type]: { 
          ...prev[type],
          text: `Error: ${message}`,
          error: true
        } 
      }));
    } finally {
      if (type === 'standard') setLoadingStandard(false);
      else setLoadingRag(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoadingStandard(true);
    setLoadingRag(true);
    setError('');
    setResponses({ standard: null, rag: null });
    setSources([]);

    // Run both queries simultaneously (Genuine Race)
    streamQuery('/query/standard', 'standard');
    streamQuery('/query/rag', 'rag');
  };

  return (
    <div className="app-shell">
      <Header page={route} />
      <main className="main-content">
        {route === 'app' ? (
          <WorkspacePage
            query={query}
            setQuery={setQuery}
            loadingStandard={loadingStandard}
            loadingRag={loadingRag}
            error={error}
            responses={responses}
            sources={sources}
            documentUploaded={documentUploaded}
            onFileUpload={handleFileUpload}
            onQuery={handleQuery}
          />
        ) : (
          <HomePage onEnterApp={() => navigate('app')} />
        )}
      </main>
    </div>
  );
}
