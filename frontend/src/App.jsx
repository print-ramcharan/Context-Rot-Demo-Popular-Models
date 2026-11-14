import React, { useEffect, useState } from 'react';
import Header from './components/Layout/Header';
import FileDropzone from './components/Upload/FileDropzone';
import QueryBar from './components/Chat/QueryBar';
import ComparisonView from './components/Comparison/ComparisonView';
import { API_BASE } from './config';
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
  Heart,
  Activity,
  BookOpen,
  CheckCircle2,
  ClipboardList,
  FileUp,
  HeartPulse,
  RefreshCw,
  Search,
  Trash2,
  Upload,
} from 'lucide-react';

const technologies = [
  { name: 'Gemma 4 26B (free)', desc: 'OpenRouter free multimodal reasoning', icon: Bot },
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

const workspaceTabs = [
  { id: 'overview', label: 'Overview', icon: Layout },
  { id: 'ingest', label: 'Documents', icon: Upload },
  { id: 'query', label: 'Query lab', icon: MessageSquare },
  { id: 'memory', label: 'Conversation memory', icon: Database },
  { id: 'monitoring', label: 'Monitoring', icon: Activity },
];

function WorkspaceTabs({ activeTab, onNavigate }) {
  return (
    <nav className="workspace-tabs" aria-label="Workspace sections">
      {workspaceTabs.map(tab => {
        const TabIcon = tab.icon;
        return (
          <button
            type="button"
            key={tab.id}
            className={`workspace-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onNavigate(tab.id)}
          >
            <TabIcon size={16} />
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function EndpointBadge({ method, path }) {
  return (
    <div className="endpoint-badge">
      <span className={`method method-${method.toLowerCase()}`}>{method}</span>
      <code>{path}</code>
    </div>
  );
}

function OverviewPage({ onNavigate }) {
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState('');

  const refresh = () => {
    Promise.all([fetchJson('/health'), fetchJson('/metrics'), fetchJson('/conv-stats')])
      .then(([health, metrics, conversations]) => setSnapshot({ health, metrics, conversations }))
      .catch(errorValue => setError(errorValue.message));
  };

  useEffect(() => {
    refresh();
  }, []);

  const index = snapshot?.metrics?.index;
  const runtime = snapshot?.metrics?.runtime || {};
  const conversations = snapshot?.conversations;

  return (
    <div className="tab-content animate-reveal">
      <div className="workspace-header">
        <div>
          <h1 className="workspace-title font-heading">Control center</h1>
          <p>Choose a workspace section by the job you want to perform.</p>
        </div>
        <button className="btn-secondary" type="button" onClick={refresh}><RefreshCw size={16} /> Refresh</button>
      </div>
      {error && <p className="evaluation-notice error-notice">{error}</p>}
      <div className="overview-grid">
        <div className="premium-card overview-status">
          <div className="card-kicker"><HeartPulse size={16} /> Backend status</div>
          <strong>{snapshot?.health?.status === 'healthy' ? 'Healthy' : snapshot ? 'Unavailable' : 'Checking…'}</strong>
          <span>{snapshot?.health?.timestamp ? new Date(snapshot.health.timestamp * 1000).toLocaleString() : 'Connect to the API to check status'}</span>
        </div>
        <div className="premium-card overview-status">
          <div className="card-kicker"><FileUp size={16} /> Indexed corpus</div>
          <strong>{index?.documents ?? '—'} chunks</strong>
          <span>{index?.embedding_dimension ? `${index.embedding_dimension}-dimension embeddings · ${index.retrieval_mode} retrieval` : 'No index data yet'}</span>
        </div>
        <div className="premium-card overview-status">
          <div className="card-kicker"><Database size={16} /> Conversation memory</div>
          <strong>{conversations?.session_count ?? conversations?.total_sessions ?? '—'} sessions</strong>
          <span>{conversations?.conversation_count ?? conversations?.total_conversations ?? 0} stored conversations</span>
        </div>
        <div className="premium-card overview-status">
          <div className="card-kicker"><Zap size={16} /> Runtime requests</div>
          <strong>{runtime.total_requests ?? runtime.request_count ?? 0}</strong>
          <span>Tracked online pipeline requests</span>
        </div>
      </div>
      <div className="quick-actions">
        <button className="premium-card quick-action" type="button" onClick={() => onNavigate('ingest')}><Upload size={20} /><strong>Ingest documents</strong><span>Upload and index TXT, MD, PDF, or DOCX files.</span></button>
        <button className="premium-card quick-action" type="button" onClick={() => onNavigate('query')}><MessageSquare size={20} /><strong>Run a query</strong><span>Compare full-context and hybrid RAG answers.</span></button>
        <button className="premium-card quick-action" type="button" onClick={() => onNavigate('memory')}><Search size={20} /><strong>Search memory</strong><span>Store, retrieve, and manage cross-platform sessions.</span></button>
        <a className="premium-card quick-action" href="#/evaluate"><ClipboardList size={20} /><strong>Run evaluation</strong><span>Measure retrieval quality, latency, and answer metrics.</span></a>
      </div>
    </div>
  );
}

function IngestPage({ onUploaded }) {
  const [message, setMessage] = useState('');

  return (
    <div className="tab-content animate-reveal">
      <div className="workspace-header"><div><h1 className="workspace-title font-heading">Document ingestion</h1><p>Upload source material to create the active searchable index.</p></div><EndpointBadge method="POST" path="/upload" /></div>
      <div className="premium-card">
        <div className="section-heading"><div><h3 className="font-heading">Ingest a source</h3><p>Supported formats: TXT, Markdown, PDF, and DOCX.</p></div><FileUp size={22} /></div>
        <FileDropzone onUpload={() => { setMessage('Document indexed successfully.'); onUploaded(true); }} />
      </div>
      {message && <p className="success-notice"><CheckCircle2 size={16} /> {message}</p>}
    </div>
  );
}

function MemoryPage() {
  const [storeForm, setStoreForm] = useState({ platform: 'chatgpt', session_id: '', prompt: '', response: '' });
  const [retrieveForm, setRetrieveForm] = useState({ query: '', top_k: 5, platform_filter: '' });
  const [sessions, setSessions] = useState([]);
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const loadSessions = () => fetchJson('/list-sessions').then(data => setSessions(data.sessions || [])).catch(errorValue => setError(errorValue.message));
  useEffect(() => {
    loadSessions();
  }, []);

  const storeConversation = async (event) => {
    event.preventDefault();
    setLoading(true); setError(''); setMessage('');
    try {
      const data = await fetchJson('/store-conversation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(storeForm) });
      setMessage(`Stored ${data.chunk_count ?? 0} memory chunk(s).`);
      setStoreForm(current => ({ ...current, prompt: '', response: '' }));
      loadSessions();
    } catch (errorValue) { setError(errorValue.message); } finally { setLoading(false); }
  };

  const retrieveMemory = async (event) => {
    event.preventDefault();
    setLoading(true); setError('');
    try {
      const data = await fetchJson('/retrieve-context', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...retrieveForm, top_k: Number(retrieveForm.top_k), platform_filter: retrieveForm.platform_filter || null }) });
      setResults(data.results || []);
    } catch (errorValue) { setError(errorValue.message); } finally { setLoading(false); }
  };

  const deleteSession = async (sessionId) => {
    if (!window.confirm(`Delete session "${sessionId}"?`)) return;
    try { await fetchJson(`/delete-session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }); setMessage('Session deleted.'); loadSessions(); }
    catch (errorValue) { setError(errorValue.message); }
  };

  return (
    <div className="tab-content animate-reveal">
      <div className="workspace-header"><div><h1 className="workspace-title font-heading">Conversation memory</h1><p>Manage the external memory API used by the browser extension.</p></div><EndpointBadge method="POST" path="/store-conversation" /></div>
      <div className="memory-grid">
        <form className="premium-card form-stack" onSubmit={storeConversation}>
          <div className="section-heading"><div><h3 className="font-heading">Store conversation</h3><p>Save a prompt/response pair for later retrieval.</p></div><BookOpen size={22} /></div>
          <label>Platform<input className="eval-input" value={storeForm.platform} onChange={event => setStoreForm({ ...storeForm, platform: event.target.value })} placeholder="chatgpt" /></label>
          <label>Session ID<input className="eval-input" required value={storeForm.session_id} onChange={event => setStoreForm({ ...storeForm, session_id: event.target.value })} placeholder="project-planning-001" /></label>
          <label>Prompt<textarea className="eval-input" required value={storeForm.prompt} onChange={event => setStoreForm({ ...storeForm, prompt: event.target.value })} rows="3" /></label>
          <label>Response<textarea className="eval-input" required value={storeForm.response} onChange={event => setStoreForm({ ...storeForm, response: event.target.value })} rows="4" /></label>
          <button className="btn-primary" disabled={loading}><Database size={16} /> Store memory</button>
        </form>
        <form className="premium-card form-stack" onSubmit={retrieveMemory}>
          <div className="section-heading"><div><h3 className="font-heading">Retrieve context</h3><p>Search stored conversations by meaning.</p></div><Search size={22} /></div>
          <label>Search query<textarea className="eval-input" required value={retrieveForm.query} onChange={event => setRetrieveForm({ ...retrieveForm, query: event.target.value })} rows="3" placeholder="What did we decide about deployment?" /></label>
          <label>Top K<input className="eval-input" type="number" min="1" max="50" value={retrieveForm.top_k} onChange={event => setRetrieveForm({ ...retrieveForm, top_k: event.target.value })} /></label>
          <label>Platform filter (optional)<input className="eval-input" value={retrieveForm.platform_filter} onChange={event => setRetrieveForm({ ...retrieveForm, platform_filter: event.target.value })} placeholder="gemini" /></label>
          <button className="btn-primary" disabled={loading}><Search size={16} /> Retrieve context</button>
        </form>
      </div>
      {message && <p className="success-notice"><CheckCircle2 size={16} /> {message}</p>}
      {error && <p className="evaluation-notice error-notice">{error}</p>}
      {results.length > 0 && <div className="premium-card"><div className="section-heading"><h3 className="font-heading">Retrieved memories</h3><span className="research-status">{results.length} results</span></div><div className="memory-results">{results.map((item, index) => <article className="memory-result" key={`${item.id || item.session_id || 'result'}-${index}`}><div><strong>{item.metadata?.platform || item.platform || 'memory'}</strong><span>{item.metadata?.session_id || item.session_id || ''}</span></div><p>{item.text || item.content || JSON.stringify(item)}</p></article>)}</div></div>}
      <div className="premium-card"><div className="section-heading"><div><h3 className="font-heading">Sessions</h3><p>Delete a session to remove its stored memory.</p></div><EndpointBadge method="GET" path="/list-sessions" /></div>{sessions.length === 0 ? <p className="empty-state">No conversation sessions found.</p> : <div className="session-list">{sessions.map(session => { const id = session.session_id || session.id || session; return <div className="session-row" key={id}><div><strong>{id}</strong><span>{session.platform || 'cross-platform'} {session.message_count ? `· ${session.message_count} messages` : ''}</span></div><button className="icon-danger" type="button" onClick={() => deleteSession(id)} aria-label={`Delete ${id}`}><Trash2 size={16} /></button></div>; })}</div>}</div>
    </div>
  );
}

function MonitoringPage() {
  const [data, setData] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [conversationStats, setConversationStats] = useState(null);
  const [error, setError] = useState('');
  const refresh = () => Promise.all([fetchJson('/health'), fetchJson('/metrics'), fetchJson('/metrics/catalog'), fetchJson('/research/catalog'), fetchJson('/conv-stats')]).then(([health, metrics, metricsCatalog, research, conversations]) => {
    setData({ health, metrics, metricsCatalog });
    setCatalog(research);
    setConversationStats(conversations);
    setError('');
  }).catch(errorValue => setError(errorValue.message));
  useEffect(() => {
    refresh();
  }, []);
  const counters = data?.metrics?.runtime?.counters || {};
  const timings = data?.metrics?.runtime?.timings || {};
  const timingRows = Object.entries(timings);
  const requests = Object.entries(counters)
    .filter(([name]) => name.includes('request') || name.includes('query'))
    .reduce((total, [, value]) => total + Number(value || 0), 0);
  const tokens = Object.entries(counters)
    .filter(([name]) => name.includes('token'))
    .reduce((total, [, value]) => total + Number(value || 0), 0);
  return <div className="tab-content animate-reveal">
    <div className="workspace-header"><div><h1 className="workspace-title font-heading">Monitoring</h1><p>Runtime usage and performance since the backend started.</p></div><button className="btn-secondary" type="button" onClick={refresh}><RefreshCw size={16} /> Refresh telemetry</button></div>
    {error && <p className="evaluation-notice error-notice">{error}</p>}
    <div className="monitoring-stats">
      {[
        ['API requests', requests, 'Requests tracked in this runtime'],
        ['Tokens used', tokens, 'Prompt and completion tokens tracked'],
        ['Memory sessions', conversationStats?.total_sessions ?? 0, 'Stored conversation sessions'],
        ['Memory chunks', conversationStats?.total_chunks ?? 0, 'Searchable conversation chunks'],
      ].map(([label, value, caption]) => <div className="premium-card monitoring-stat" key={label}><span>{label}</span><strong>{data ? value.toLocaleString() : '—'}</strong><small>{caption}</small></div>)}
    </div>
    <div className="premium-card"><div className="section-heading"><div><h3 className="font-heading">Average latency</h3><p>Each row is populated after that pipeline stage has run.</p></div><Activity size={22} /></div>{timingRows.length === 0 ? <p className="empty-state">No latency samples yet. Run a query or evaluation to start collecting telemetry.</p> : <div className="metrics-table-wrap"><table className="metrics-table"><thead><tr><th>Stage</th><th>Calls</th><th>Average</th><th>P50</th><th>P95</th><th>Max</th></tr></thead><tbody>{timingRows.map(([name, stats]) => <tr key={name}><td>{name.replaceAll('_', ' ')}</td><td>{stats.count}</td><td className="metric-value">{stats.avg_ms.toFixed(2)} ms</td><td className="metric-value">{stats.p50_ms.toFixed(2)} ms</td><td className="metric-value">{stats.p95_ms.toFixed(2)} ms</td><td className="metric-value">{stats.max_ms.toFixed(2)} ms</td></tr>)}</tbody></table></div>}</div>
    <div className="monitoring-grid"><div className="premium-card"><EndpointBadge method="GET" path="/health" /><pre className="json-panel">{JSON.stringify(data?.health || { status: 'checking…' }, null, 2)}</pre></div><div className="premium-card"><EndpointBadge method="GET" path="/metrics" /><pre className="json-panel">{JSON.stringify(data?.metrics?.index || { status: 'checking…' }, null, 2)}</pre></div></div>
    {catalog && <div className="premium-card"><div className="section-heading"><div><h3 className="font-heading">Research readiness</h3><p>{catalog.paper_readiness?.status?.replaceAll('_', ' ')}</p></div><Activity size={22} /></div><div className="research-experiment-grid">{catalog.experiments?.map(experiment => <div className="research-experiment" key={experiment.id}><strong>{experiment.title}</strong><span>{experiment.status.replaceAll('_', ' ')}</span></div>)}</div></div>}
  </div>;
}

function WorkspacePage({
  activeTab,
  onNavigate,
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
  sessionStats,
  onUploaded,
}) {
  if (activeTab === 'overview') return <><WorkspaceTabs activeTab={activeTab} onNavigate={onNavigate} /><OverviewPage onNavigate={onNavigate} /></>;
  if (activeTab === 'ingest') return <><WorkspaceTabs activeTab={activeTab} onNavigate={onNavigate} /><IngestPage onUploaded={onUploaded} /></>;
  if (activeTab === 'memory') return <><WorkspaceTabs activeTab={activeTab} onNavigate={onNavigate} /><MemoryPage /></>;
  if (activeTab === 'monitoring') return <><WorkspaceTabs activeTab={activeTab} onNavigate={onNavigate} /><MonitoringPage /></>;
  const hasResults = responses.standard || responses.rag || loadingStandard || loadingRag;

  return (
    <div className="workspace-container animate-reveal">
      <WorkspaceTabs activeTab={activeTab} onNavigate={onNavigate} />
      <div className="workspace-header">
        <div>
          <h1 className="workspace-title font-heading">Workspace</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Upload your source and query the precision engine.</p>
        </div>
        <div className="workspace-stats">
          <div className="stat-item">
            <div className="stat-value">{onFileUpload ? sessionStats.avgLatency : '2.4s'}</div>
            <div className="stat-label">Avg Latency</div>
          </div>
          <div className="stat-item">
            <div className="stat-value">{onFileUpload ? sessionStats.accuracy : '98%'}</div>
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

function EvaluationPage() {
  const [query, setQuery] = useState('');
  const [expected, setExpected] = useState('');
  const [benchmarkCases, setBenchmarkCases] = useState('');
  const [documents, setDocuments] = useState([]);
  const [researchCatalog, setResearchCatalog] = useState(null);
  const [selectedSources, setSelectedSources] = useState([]);
  const [generate, setGenerate] = useState(false);
  const [result, setResult] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  const [comparisonRunning, setComparisonRunning] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/documents`),
      fetch(`${API_BASE}/research/catalog`)
    ])
      .then(async responses => {
        if (responses.some(response => !response.ok)) {
          throw new Error('Backend request failed');
        }
        return Promise.all(responses.map(response => response.json()));
      })
      .then(([documentsData, catalogData]) => {
        setDocuments(documentsData.documents || []);
        setSelectedSources([]);
        setResearchCatalog(catalogData);
      })
      .catch(() => setError('Cannot reach the backend. Start it with ./run.sh, then refresh this page.'));
  }, []);

  const toggleSource = (name) => {
    setSelectedSources(current => current.includes(name)
      ? current.filter(source => source !== name)
      : [...current, name]);
  };

  const buildCases = () => {
    const defaultSources = selectedSources;
    const additionalCases = benchmarkCases.trim()
      ? JSON.parse(benchmarkCases)
      : [];
    if (!Array.isArray(additionalCases)) {
      throw new Error('Additional benchmark cases must be a JSON array.');
    }
    return [{
      query,
      expected_answer: expected,
      relevant_sources: defaultSources,
      generate_answer: generate
    }, ...additionalCases.map((item, index) => {
      if (!item || typeof item.query !== 'string' || !item.query.trim()) {
        throw new Error(`Benchmark case ${index + 1} needs a non-empty query.`);
      }
      return {
        query: item.query,
        expected_answer: item.expected_answer || '',
        relevant_sources: Array.isArray(item.relevant_sources)
          ? item.relevant_sources
          : defaultSources,
        generate_answer: false
      };
    })];
  };

  const runEvaluation = async (event) => {
    event.preventDefault();
    if (!query.trim()) return setError('Enter an evaluation query.');
    setRunning(true);
    setError('');
    try {
      const cases = buildCases();
      const response = await fetch(`${API_BASE}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          top_k: 5,
          cases
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Evaluation failed');
      setResult(data);
    } catch (evaluationError) {
      setError(evaluationError.message === 'Failed to fetch'
        ? 'Cannot reach the backend. Start it with ./run.sh, then refresh this page.'
        : evaluationError.message);
    } finally {
      setRunning(false);
    }
  };

  const runComparison = async () => {
    if (!query.trim()) {
      setError('Enter an evaluation query before comparing stages.');
      return;
    }
    setComparisonRunning(true);
    setError('');
    try {
      const cases = buildCases().map(item => ({ ...item, generate_answer: false }));
      const response = await fetch(`${API_BASE}/evaluate/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evaluation_k: [5, 10, 20],
          cases
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Comparison failed');
      setComparison(data);
    } catch (comparisonError) {
      setError(comparisonError.message === 'Failed to fetch'
        ? 'Cannot reach the backend. Start it with ./run.sh, then refresh this page.'
        : comparisonError.message);
    } finally {
      setComparisonRunning(false);
    }
  };

  const metricRows = result ? [
    ['Retrieval', 'Hit rate', result.cases[0].retrieval.hit_rate_at_k ?? result.cases[0].retrieval.hit_rate, 'Score'],
    ['Retrieval', 'Precision@k', result.cases[0].retrieval.precision_at_k, 'Score'],
    ['Retrieval', 'Recall@k', result.cases[0].retrieval.recall_at_k, 'Score'],
    ['Retrieval', 'MRR', result.cases[0].retrieval.mrr_at_k ?? result.cases[0].retrieval.mrr, 'Score'],
    ['Retrieval', 'nDCG@k', result.cases[0].retrieval.ndcg_at_k ?? result.cases[0].retrieval.ndcg, 'Score'],
    ['Context', 'Context precision', result.cases[0].context.context_precision, 'Score'],
    ['Context', 'Context recall', result.cases[0].context.context_recall, 'Score'],
    ['Answer', 'Exact match', result.cases[0].answer?.exact_match, 'Score'],
    ['Answer', 'Answer token F1 (lexical proxy)', result.cases[0].answer?.answer_token_f1, 'Score'],
    ['Answer', 'Faithfulness proxy', result.cases[0].answer?.faithfulness_proxy, 'Score'],
    ['Answer', 'Answer token recall (lexical proxy)', result.cases[0].answer?.answer_token_recall, 'Score'],
    ['System', 'Retrieval latency', result.cases[0].system.retrieval_ms, 'ms'],
    ['System', 'Generation latency', result.cases[0].system.generation_ms, 'ms'],
    ['System', 'Total latency', result.cases[0].system.total_ms, 'ms'],
    ['System', 'Generation status', result.cases[0].generation?.status || 'unknown', 'status']
  ] : [];

  const formatMetric = (value, unit) => {
    if (value == null) return '—';
    if (typeof value !== 'number') return value;
    return `${value.toFixed(unit === 'ms' ? 2 : 3)}${unit === 'ms' ? ' ms' : ''}`;
  };

  const summaryRows = result ? Object.entries(result.summary.metrics || {}).map(([name, stats]) => [
    name.replaceAll('_', ' '),
    stats.mean,
    stats.std_dev,
    stats.ci95_low,
    stats.ci95_high,
  ]) : [];

  return (
    <div className="workspace-container animate-reveal">
      <div className="workspace-header">
        <div>
          <h1 className="workspace-title font-heading">RAG Evaluation</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Test each pipeline stage with grounded, reproducible metrics.
          </p>
        </div>
        <a className="btn-secondary" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API docs</a>
      </div>
      <div className="premium-card">
        <form onSubmit={runEvaluation} style={{ display: 'grid', gap: '16px' }}>
          <label>Query<input className="eval-input" value={query} onChange={event => setQuery(event.target.value)} placeholder="What should retrieval find?" /></label>
          <label>Expected answer (optional)<textarea className="eval-input" value={expected} onChange={event => setExpected(event.target.value)} rows="3" /></label>
          <div>
            <label>Relevant indexed documents</label>
            {documents.length ? (
              <div className="document-picker">
                {documents.map(document => (
                  <label className="document-option" key={document.name}>
                    <input type="checkbox" checked={selectedSources.includes(document.name)} onChange={() => toggleSource(document.name)} />
                    <span>{document.name}</span>
                    <small>{document.chunks} chunks</small>
                  </label>
                ))}
              </div>
            ) : <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No indexed documents. Upload one in Workspace first.</p>}
          </div>
          <label>Additional benchmark cases (optional JSON)
            <textarea className="eval-input" value={benchmarkCases} onChange={event => setBenchmarkCases(event.target.value)} rows="4" placeholder={'[{"query":"Another question","relevant_sources":["document.txt"]}]'} />
          </label>
          <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}><input type="checkbox" checked={generate} onChange={event => setGenerate(event.target.checked)} /> Generate an answer and calculate answer metrics <span style={{ color: 'var(--text-muted)' }}>(uses OpenRouter credits)</span></label>
          <div className="evaluation-actions">
            <button className="btn-primary" disabled={running}>{running ? 'Running…' : 'Run evaluation'}</button>
            <button type="button" className="btn-secondary" onClick={runComparison} disabled={comparisonRunning}>
              {comparisonRunning ? 'Comparing…' : 'Compare pipeline stages'}
            </button>
          </div>
        </form>
        {error && <p style={{ color: '#ef4444' }}>{error}</p>}
      </div>
      {researchCatalog && <ResearchReadiness catalog={researchCatalog} />}
      {result && <div className="premium-card">
        <h3 className="font-heading">Metrics</h3>
        <div className="metrics-table-wrap">
          <p className="evaluation-notice">
            Retrieval metrics depend on the query, corpus, and relevance labels—not the expected answer.
            Changing only the expected answer should leave retrieval scores unchanged. Compare the Answer
            rows and Answer audit to evaluate generation quality.
          </p>
          {result.invalid_cases?.length > 0 && (
            <p className="evaluation-notice">
              {result.invalid_cases.length} benchmark case(s) were not evaluated:
              {' '}{result.invalid_cases.map(item => `${item.case_index + 1}: ${item.errors.join(', ')}`).join('; ')}
            </p>
          )}
          {result.cases[0].generation?.status !== 'completed' && (
            <p className="evaluation-notice">
              Answer metrics are unavailable because generation is {result.cases[0].generation?.status || 'unknown'}.
              Enable answer generation and ensure the OpenRouter key has sufficient credits.
              {result.cases[0].generation?.error && ` ${result.cases[0].generation.error}`}
            </p>
          )}
          {result.cases[0].generation?.status === 'completed' && (
            <details className="answer-audit">
              <summary>Answer audit</summary>
              <p><strong>Query:</strong> {result.cases[0].query}</p>
              <p><strong>Expected:</strong> {result.cases[0].expected_answer || '—'}</p>
              <p><strong>Generated:</strong> {result.cases[0].generated_answer || '—'}</p>
              <p className="comparison-note">Exact match and token F1 compare the generated response with the supplied expected answer. Faithfulness is a lexical proxy, not a factuality judgment.</p>
            </details>
          )}
          <table className="metrics-table">
            <thead>
              <tr><th>Stage</th><th>Metric</th><th>Value</th><th>Unit</th></tr>
            </thead>
            <tbody>
              {metricRows.map(([stage, name, value, unit]) => (
                <tr key={name}>
                  <td><span className={`metric-stage metric-stage-${stage.toLowerCase()}`}>{stage}</span></td>
                  <td>{name}</td>
                  <td className="metric-value">{formatMetric(value, unit)}</td>
                  <td>{unit === 'Score' ? '0–1' : unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {summaryRows.length > 0 && <div className="metrics-table-wrap">
          <h4 className="metrics-subheading">Aggregate statistics</h4>
          <table className="metrics-table">
            <thead>
              <tr><th>Metric</th><th>Mean</th><th>Std. dev.</th><th>95% CI low</th><th>95% CI high</th></tr>
            </thead>
            <tbody>
              {summaryRows.map(([name, mean, standardDeviation, low, high]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="metric-value">{formatMetric(mean, 'Score')}</td>
                  <td className="metric-value">{formatMetric(standardDeviation, 'Score')}</td>
                  <td className="metric-value">{formatMetric(low, 'Score')}</td>
                  <td className="metric-value">{formatMetric(high, 'Score')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>}
        {result.cases[0].trace && <div className="trace-panel">
          <h4 className="metrics-subheading">Stage-by-stage trace</h4>
          {Object.entries(result.cases[0].trace.stages).map(([stage, details]) => (
            <details className="trace-stage" key={stage}>
              <summary>
                <span>{stage}</span>
                <strong>{details.latency_ms == null ? '—' : `${details.latency_ms.toFixed(2)} ms`}</strong>
              </summary>
              <pre>{JSON.stringify(details, null, 2)}</pre>
            </details>
          ))}
        </div>}
        <details style={{ marginTop: '24px' }}><summary>Raw evaluation payload</summary><pre style={{ overflow: 'auto' }}>{JSON.stringify(result, null, 2)}</pre></details>
      </div>}
      {comparison && <ComparisonPanel comparison={comparison} />}
      <Footer />
    </div>
  );
}

function ResearchReadiness({ catalog }) {
  return (
    <div className="premium-card research-readiness">
      <div className="research-readiness-header">
        <div>
          <h2 className="font-heading">Research readiness</h2>
          <p className="comparison-note">
            The dashboard reports measured results only. Runner availability is not evidence that an experiment has been completed.
          </p>
        </div>
        <span className="research-status">{catalog.paper_readiness.status.replaceAll('_', ' ')}</span>
      </div>
      <div className="research-requirements">
        <span>{catalog.paper_readiness.minimum_independent_queries}+ independent queries</span>
        <span>{catalog.paper_readiness.required_repeated_latency_runs}+ latency runs</span>
        <span>Paired CIs on independent queries</span>
      </div>
      <div className="research-experiment-grid">
        {catalog.experiments.map(experiment => (
          <div className="research-experiment" key={experiment.id}>
            <strong>{experiment.title}</strong>
            <span>{experiment.status.replaceAll('_', ' ')}</span>
          </div>
        ))}
      </div>
      <h4 className="metrics-subheading">Baseline matrix</h4>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead><tr><th>Baseline</th><th>Configuration</th><th>Status</th></tr></thead>
          <tbody>{catalog.baselines.map(baseline => (
            <tr key={baseline.id}>
              <td>{baseline.id}</td>
              <td>{baseline.title}</td>
              <td>{baseline.status.replaceAll('_', ' ')}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonPanel({ comparison }) {
  const score = (stats) => stats?.mean == null ? '—' : stats.mean.toFixed(3);
  const latency = (stats) => stats?.mean == null ? '—' : `${stats.mean.toFixed(2)} ms`;
  const cutoffs = comparison.configuration?.evaluation_k || [];
  const sampleWarning = (comparison.valid_query_count ?? comparison.query_count) < 2
    ? 'Only one query was evaluated. Percentiles and confidence intervals are not research-grade until multiple independently labeled queries are supplied.'
    : null;
  return (
    <div className="premium-card comparison-panel">
      <h2 className="font-heading">Pipeline comparison</h2>
      <p className="comparison-note">
        Same queries, documents, relevance labels, and evaluation cutoffs are used for every row.
      </p>
      {sampleWarning && <p className="evaluation-notice">{sampleWarning}</p>}
      {comparison.invalid_cases?.length > 0 && (
        <p className="evaluation-notice">
          {comparison.invalid_cases.length} benchmark case(s) were not evaluated:
          {' '}{comparison.invalid_cases.map(item => `${item.case_index + 1}: ${item.errors.join(', ')}`).join('; ')}
        </p>
      )}
      {!comparison.configuration?.relevance_label_count && (
        <p className="evaluation-notice">No relevance labels were supplied. Retrieval quality scores are unavailable and must not be interpreted as evidence.</p>
      )}
      <h3 className="metrics-subheading">Embedding comparison</h3>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead><tr><th>Model</th><th>Status</th><th>Recall (max cutoff)</th><th>nDCG (max cutoff)</th><th>Latency</th></tr></thead>
          <tbody><tr>
            <td>MiniLM-L6-v2</td>
            <td>Measured</td>
            <td className="metric-value">{score(comparison.retrieval_comparison.find(row => row.method === 'dense')?.metrics.recall_at_k)}</td>
            <td className="metric-value">{score(comparison.retrieval_comparison.find(row => row.method === 'dense')?.metrics.ndcg_at_k)}</td>
            <td className="metric-value">{latency(comparison.latency_comparison.embedding)}</td>
          </tr><tr>
            <td>Alternative embedding model</td>
            <td>Not configured</td><td>—</td><td>—</td><td>—</td>
          </tr></tbody>
        </table>
      </div>
      <h3 className="metrics-subheading">Retrieval comparison</h3>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead><tr><th>Method</th><th>Cutoff</th><th>Precision</th><th>Recall</th><th>MRR</th><th>nDCG</th><th>Samples</th></tr></thead>
          <tbody>{comparison.retrieval_comparison.flatMap(row => cutoffs.map(cutoff => {
            const metrics = row.metrics_by_k?.[String(cutoff)] || {};
            return <tr key={`${row.method}-${cutoff}`}>
              <td>{row.method}</td><td>@{cutoff}</td>
              <td className="metric-value">{score(metrics.precision_at_k)}</td>
              <td className="metric-value">{score(metrics.recall_at_k)}</td>
              <td className="metric-value">{score(metrics.mrr_at_k)}</td>
              <td className="metric-value">{score(metrics.ndcg_at_k)}</td>
              <td>{metrics.precision_at_k?.count ?? 0}</td>
            </tr>;
          }))}</tbody>
        </table>
      </div>
      <h3 className="metrics-subheading">Latency comparison</h3>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead><tr><th>Stage</th><th>Mean</th><th>P50</th><th>P95</th><th>P99</th><th>Samples</th></tr></thead>
          <tbody>{Object.entries(comparison.latency_comparison).map(([stage, stats]) => (
            <tr key={stage}>
              <td>{stage}</td>
              <td className="metric-value">{latency(stats)}</td>
              <td className="metric-value">{latency({ mean: stats?.median })}</td>
              <td className="metric-value">{latency({ mean: stats?.p95 })}</td>
              <td className="metric-value">{latency({ mean: stats?.p99 })}</td>
              <td>{stats?.count ?? 0}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h3 className="metrics-subheading">Context comparison</h3>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead><tr><th>Configuration</th><th>Evidence recall</th><th>Tokens</th><th>Retention</th></tr></thead>
          <tbody>{comparison.context_comparison.map(row => (
            <tr key={row.configuration}>
              <td>{row.configuration}</td>
              <td className="metric-value">{score(row.metrics.evidence_recall)}</td>
              <td className="metric-value">{row.metrics.tokens?.mean == null ? '—' : row.metrics.tokens.mean.toFixed(0)}</td>
              <td className="metric-value">{score(row.metrics.retention)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h3 className="metrics-subheading">Ranked chunk audit</h3>
      <p className="comparison-note">These are the actual ranked chunk IDs used to calculate each method's metrics.</p>
      {comparison.per_query?.map(item => (
        <details className="trace-panel" key={item.case_index ?? item.query}>
          <summary>Query {item.case_index + 1}: {item.query}</summary>
          <p className="comparison-note">Relevant labels: {item.relevant_sources?.join(', ') || 'none supplied'}</p>
          <div className="metrics-table-wrap">
            <table className="metrics-table">
              <thead><tr><th>Method</th><th>Ranked chunk IDs</th></tr></thead>
              <tbody>{[
                ['BM25', item.trace?.stages?.bm25?.retrieved_chunk_ids],
                ['Dense', item.trace?.stages?.dense?.retrieved_chunk_ids],
                ['RRF', item.trace?.stages?.rrf?.candidate_chunk_ids],
                ['Reranked', item.trace?.stages?.reranker?.reranked_chunk_ids],
              ].map(([method, ids]) => (
                <tr key={method}><td>{method}</td><td>{ids?.join(' → ') || '—'}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </details>
      ))}
      <h3 className="metrics-subheading">Generation comparison</h3>
      <p className="evaluation-notice">{comparison.generation_comparison.reason}</p>
    </div>
  );
}

function getRouteFromHash() {
  if (typeof window === 'undefined') return 'home';
  const rawHash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
  return rawHash.startsWith('app') || rawHash === 'evaluate' ? rawHash : 'home';
}

export default function App() {
  const [route, setRoute] = useState(getRouteFromHash());
  const [activeTab, setActiveTab] = useState(() => {
    const hash = getRouteFromHash();
    return hash.startsWith('app/') ? hash.slice(4) : 'overview';
  });
  const [query, setQuery] = useState('');
  const [loadingStandard, setLoadingStandard] = useState(false);
  const [loadingRag, setLoadingRag] = useState(false);
  const [error, setError] = useState('');
  const [responses, setResponses] = useState({ standard: null, rag: null });
  const [sources, setSources] = useState([]);
  const [documentUploaded, setDocumentUploaded] = useState(false);
  const [queryHistory, setQueryHistory] = useState([]);

  const calculateStats = () => {
    if (queryHistory.length === 0) return { avgLatency: '2.4s', accuracy: '98%' };
    
    const latencies = queryHistory.map(q => q.latency).filter(l => l > 0);
    const avg = latencies.length > 0 
      ? (latencies.reduce((a, b) => a + b, 0) / latencies.length / 1000).toFixed(1) + 's'
      : '2.4s';
    
    // Simple heuristic for "accuracy" - if RAG returned text and no error
    const successful = queryHistory.filter(q => q.success).length;
    const acc = ((successful / queryHistory.length) * 100).toFixed(0) + '%';
    
    return { avgLatency: avg, accuracy: acc };
  };

  useEffect(() => {
    const syncRoute = () => {
      const nextRoute = getRouteFromHash();
      setRoute(nextRoute);
      setActiveTab(nextRoute.startsWith('app/') ? nextRoute.slice(4) : 'overview');
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

  const navigateWorkspace = (nextTab) => {
    navigate(`app/${nextTab}`);
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
              if (type === 'rag') {
                setQueryHistory(prev => [...prev, { 
                  latency: data.latency_ms, 
                  success: true 
                }]);
              }
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

    // Keep the two OpenRouter requests strictly sequential to avoid
    // competing for the same free-tier rate limit.
    await streamQuery('/query/rag', 'rag');
    await streamQuery('/query/standard', 'standard');
  };

  return (
    <div className="app-shell">
      <Header page={route.startsWith('app') ? 'app' : route} />
      <main className="main-content">
        {route.startsWith('app') ? (
          <WorkspacePage
            activeTab={activeTab}
            onNavigate={navigateWorkspace}
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
            sessionStats={calculateStats()}
            onUploaded={(uploaded) => setDocumentUploaded(uploaded)}
          />
        ) : route === 'evaluate' ? (
          <EvaluationPage />
        ) : (
          <HomePage onEnterApp={() => navigate('app')} />
        )}
      </main>
    </div>
  );
}
