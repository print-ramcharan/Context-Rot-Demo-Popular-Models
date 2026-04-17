import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle2, Loader2, AlertCircle, FileText, RefreshCw } from 'lucide-react';

export default function FileDropzone({ onUpload }) {
  const [status, setStatus] = useState('idle');
  const [fileName, setFileName] = useState('');
  const [uploadInfo, setUploadInfo] = useState(null);
  const [progress, setProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    try {
      setStatus('uploading');
      setFileName(file.name);
      setProgress(0);

      const formData = new FormData();
      formData.append('file', file);

      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await axios.post('http://127.0.0.1:8000/upload?clear_existing=true', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      clearInterval(progressInterval);
      setProgress(100);
      setStatus('success');
      setUploadInfo({
        filename: file.name,
        chunks: response.data.chunks_created || 0,
        embeddings: response.data.embeddings_stored || 0,
      });
      onUpload?.(file);
    } catch (err) {
      console.error('Upload error:', err);
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
  }, [onUpload]);

  const handleFileChange = (e) => { handleFile(e.target.files?.[0]); };

  const handleDrag = useCallback((e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  }, [handleFile]);

  const handleReupload = () => {
    setStatus('idle'); setFileName(''); setUploadInfo(null); setProgress(0);
  };

  /* ── SUCCESS STATE ── */
  if (status === 'success' && uploadInfo) {
    return (
      <div className="animate-fade-up" style={{
        borderRadius: '14px',
        border: '1px solid rgba(16, 185, 129, 0.3)',
        background: 'var(--bg-surface)',
        padding: '24px',
        boxShadow: '0 0 0 4px rgba(16, 185, 129, 0.06)',
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
          <div style={{
            width: '44px', height: '44px', flexShrink: 0,
            borderRadius: '12px',
            background: 'rgba(16, 185, 129, 0.12)',
            border: '1px solid rgba(16, 185, 129, 0.25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <CheckCircle2 size={22} style={{ color: 'var(--emerald-400)' }} />
          </div>

          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '13px', fontWeight: 600, color: 'var(--emerald-400)', marginBottom: '4px' }}>
              Document indexed successfully
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '16px' }}>
              <FileText size={13} style={{ color: 'var(--text-muted)' }} />
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500 }}>
                {uploadInfo.filename}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '16px' }}>
              {[
                { label: 'Chunks Created', value: uploadInfo.chunks },
                { label: 'Embeddings Stored', value: uploadInfo.embeddings },
              ].map(({ label, value }) => (
                <div key={label} style={{
                  borderRadius: '10px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-card)',
                  padding: '12px 14px',
                }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>{label}</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--emerald-400)', fontFamily: "'JetBrains Mono', monospace" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            <button onClick={handleReupload} style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              fontSize: '12px', color: 'var(--text-muted)', cursor: 'pointer',
              background: 'none', border: 'none', padding: 0,
              fontFamily: 'inherit',
            }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--text-secondary)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <RefreshCw size={12} />
              Replace document
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ── UPLOAD ZONE ── */
  const isError = status === 'error';
  const isUploading = status === 'uploading';

  return (
    <div style={{ width: '100%' }}>
      <label
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          minHeight: '180px',
          borderRadius: '14px',
          border: `2px dashed ${dragActive ? 'var(--purple-500)' :
              isError ? 'var(--red-400)' :
                isUploading ? 'var(--purple-400)' :
                  'var(--border-strong)'
            }`,
          background: dragActive ? 'rgba(139,92,246,0.05)' : 'var(--bg-surface)',
          cursor: isUploading ? 'default' : 'pointer',
          textAlign: 'center',
          padding: '32px 24px',
          transition: 'border-color 0.2s ease, background 0.2s ease',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <input
          type="file"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          accept=".pdf,.txt,.docx,.md"
          disabled={isUploading}
        />

        {/* Icon */}
        <div style={{ marginBottom: '12px' }}>
          {isUploading && (
            <div style={{ position: 'relative', display: 'inline-flex' }}>
              <Loader2 size={36} className="animate-spin-slow" style={{ color: 'var(--purple-400)' }} />
            </div>
          )}
          {isError && <AlertCircle size={36} style={{ color: 'var(--red-400)' }} />}
          {!isUploading && !isError && (
            <UploadCloud size={36} style={{
              color: dragActive ? 'var(--purple-400)' : 'var(--text-muted)',
              transition: 'color 0.2s ease',
            }} />
          )}
        </div>

        {/* Text */}
        <p style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
          {isUploading ? (
            progress >= 90 
              ? `Finalizing index for ${fileName}...` 
              : `Indexing ${fileName}…`
          ) : (
            isError ? 'Upload failed — please try again' : (dragActive ? 'Drop to upload' : 'Drop a document, or click to browse')
          )}
        </p>
        
        {isUploading && progress >= 90 && (
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px', animation: 'pulse 2s infinite' }}>
            Generating embeddings... This can take up to a minute for massive files like War and Peace.
          </p>
        )}

        {!isUploading && !isError && (
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            PDF, DOCX, TXT, Markdown · Max 10 MB
          </p>
        )}

        {/* Progress bar */}
        {isUploading && (
          <div style={{ marginTop: '20px', width: '240px' }}>
            <div style={{
              height: '4px', width: '100%',
              borderRadius: '9999px',
              background: 'var(--bg-hover)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${progress}%`,
                borderRadius: '9999px',
                background: 'linear-gradient(90deg, var(--purple-600), var(--purple-400))',
                transition: 'width 0.3s ease',
              }} />
            </div>
            <p style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>
              {progress}%
            </p>
          </div>
        )}
      </label>
    </div>
  );
}
