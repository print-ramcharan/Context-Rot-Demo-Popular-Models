import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Layout/Header';
import FileDropzone from './components/Upload/FileDropzone';
import QueryBar from './components/Chat/QueryBar';
import ComparisonView from './components/Comparison/ComparisonView';
import { AlertCircle } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  // State management
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [responses, setResponses] = useState({ 
    standard: null, 
    rag: null 
  });
  const [sources, setSources] = useState([]);
  const [documentUploaded, setDocumentUploaded] = useState(false);

  // Handle file upload success
  const handleFileUpload = (file) => {
    console.log('Document uploaded:', file.name);
    setDocumentUploaded(true);
    setError(''); // Clear any previous errors
  };

  // Handle query submission
  const handleQuery = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError('');
    setResponses({ standard: null, rag: null });
    setSources([]);

    try {
      const { data } = await axios.post(`${API_BASE}/query`, {
        user_query: query,
      });

      console.log('Query response:', data);

      if (data.status !== 'success') {
        throw new Error(data.detail || 'Query failed');
      }

      // Set responses
      setResponses({
        standard: data.responses?.standard || { text: 'No response available' },
        rag: data.responses?.rag || { text: 'No response available' },
      });

      // Set sources if available
      if (data.sources && Array.isArray(data.sources)) {
        setSources(data.sources);
      } else if (data.responses?.rag?.sources) {
        setSources(data.responses.rag.sources);
      }

    } catch (err) {
      console.error('Query error:', err);
      const message = err?.response?.data?.detail 
        || err?.response?.data?.message 
        || err.message 
        || 'An error occurred while processing your query';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white">
      <Header />

      <main className="mx-auto max-w-7xl px-6 py-12 space-y-12">
        
        {/* Step 1: Upload */}
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-sm font-bold text-emerald-400">
              1
            </div>
            <h2 className="text-sm font-bold tracking-wide text-gray-300 uppercase">
              Upload Document Sources
            </h2>
          </div>
          <FileDropzone onUpload={handleFileUpload} />
        </section>

        {/* Step 2: Query */}
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg border text-sm font-bold transition-colors ${
              documentUploaded 
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : 'border-[#1a1a1a] bg-[#0b0b0b] text-gray-600'
            }`}>
              2
            </div>
            <h2 className={`text-sm font-bold tracking-wide uppercase transition-colors ${
              documentUploaded ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Ask Your Question
            </h2>
          </div>
          
          <QueryBar
            value={query}
            onChange={setQuery}
            onSubmit={handleQuery}
            loading={loading}
          />

          {/* Error Display */}
          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 animate-in fade-in slide-in-from-top-2 duration-300">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 flex-shrink-0 text-red-400 mt-0.5" />
                <div>
                  <h4 className="text-sm font-semibold text-red-300">Error</h4>
                  <p className="mt-1 text-sm text-red-200/80">{error}</p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Step 3: Results */}
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg border text-sm font-bold transition-colors ${
              responses.standard || responses.rag || loading
                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                : 'border-[#1a1a1a] bg-[#0b0b0b] text-gray-600'
            }`}>
              3
            </div>
            <h2 className={`text-sm font-bold tracking-wide uppercase transition-colors ${
              responses.standard || responses.rag || loading ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Compare Responses
            </h2>
          </div>

          {(loading || responses.standard || responses.rag) ? (
            <ComparisonView
              standard={responses.standard}
              rag={responses.rag}
              loading={loading}
              sources={sources}
            />
          ) : (
            <div className="rounded-2xl border-2 border-dashed border-[#1a1a1a] bg-[#0b0b0b] px-6 py-16 text-center">
              <p className="text-sm text-gray-500">
                Upload a document and ask a question to see the comparison
              </p>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1a1a1a] py-6">
        <div className="mx-auto max-w-7xl px-6 text-center text-xs text-gray-500">
          <p>
            Built with React · FastAPI · Gemini · FAISS
            <span className="mx-2">·</span>
            <a 
              href="https://github.com" 
              className="text-emerald-400 hover:text-emerald-300 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Source
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}