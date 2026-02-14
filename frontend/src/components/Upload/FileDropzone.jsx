import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle2, Loader2, AlertCircle, FileText } from 'lucide-react';

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

      const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      clearInterval(progressInterval);
      setProgress(100);
      setStatus('success');
      
      // Keep upload info permanently - DON'T reset!
      setUploadInfo({
        filename: file.name,
        chunks: response.data.chunks_created || 0,
        embeddings: response.data.embeddings_stored || 0
      });
      
      onUpload?.(file);

    } catch (err) {
      console.error('Upload error:', err);
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
  }, [onUpload]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    handleFile(file);
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  }, [handleFile]);

  const handleReupload = () => {
    setStatus('idle');
    setFileName('');
    setUploadInfo(null);
    setProgress(0);
  };

  // ✅ SUCCESS STATE - Shows permanently!
  if (status === 'success' && uploadInfo) {
    return (
      <div className="w-full rounded-2xl border-2 border-emerald-500/50 bg-emerald-500/5 px-6 py-8">
        <div className="flex items-start gap-4">
          <div className="relative flex-shrink-0">
            <CheckCircle2 className="h-12 w-12 text-emerald-400" />
            <div className="absolute inset-0 rounded-full bg-emerald-400/20 blur-xl" />
          </div>
          
          <div className="flex-1">
            <h3 className="text-lg font-bold text-emerald-400 mb-1">
              ✓ Document Indexed Successfully
            </h3>
            
            <div className="flex items-center gap-2 text-sm text-gray-400 mb-3">
              <FileText className="h-4 w-4" />
              <span className="font-medium text-white">{uploadInfo.filename}</span>
            </div>
            
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="rounded-lg border border-emerald-500/20 bg-[#0b0b0b] px-3 py-2">
                <div className="text-xs text-gray-500">Chunks Created</div>
                <div className="text-lg font-bold text-emerald-400">{uploadInfo.chunks}</div>
              </div>
              <div className="rounded-lg border border-emerald-500/20 bg-[#0b0b0b] px-3 py-2">
                <div className="text-xs text-gray-500">Embeddings Stored</div>
                <div className="text-lg font-bold text-emerald-400">{uploadInfo.embeddings}</div>
              </div>
            </div>
            
            <button
              onClick={handleReupload}
              className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors underline"
            >
              Upload a different document
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Regular upload zone
  return (
    <div className="w-full">
      <label
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`
          group relative flex w-full cursor-pointer flex-col items-center justify-center 
          rounded-2xl border-2 border-dashed transition-all duration-300
          ${dragActive 
            ? 'border-emerald-500 bg-emerald-500/5' 
            : status === 'error'
            ? 'border-red-500/50 bg-red-500/5'
            : 'border-[#1a1a1a] bg-[#0b0b0b] hover:border-emerald-500/30 hover:bg-[#0f0f0f]'
          }
          px-6 py-12 text-center
        `}
      >
        <input
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept=".pdf,.txt,.docx,.md"
          disabled={status === 'uploading'}
        />

        <div className="mb-4">
          {status === 'uploading' && (
            <div className="relative">
              <Loader2 className="h-12 w-12 animate-spin text-emerald-400" />
              <div className="absolute inset-0 animate-pulse rounded-full bg-emerald-400/20 blur-xl" />
            </div>
          )}
          {status === 'error' && (
            <AlertCircle className="h-12 w-12 text-red-400" />
          )}
          {status === 'idle' && (
            <UploadCloud className={`h-12 w-12 transition-colors ${
              dragActive ? 'text-emerald-400' : 'text-gray-500 group-hover:text-emerald-400'
            }`} />
          )}
        </div>

        <div className="space-y-2">
          <p className="text-sm font-semibold text-white">
            {status === 'uploading' && `Indexing ${fileName}...`}
            {status === 'error' && 'Upload failed — please try again'}
            {status === 'idle' && (dragActive ? 'Drop your document here' : 'Drop a document or click to upload')}
          </p>
          
          {status === 'idle' && (
            <p className="text-xs text-gray-500">
              Supports PDF, DOCX, TXT, and Markdown · Max 10MB
            </p>
          )}

          {status === 'uploading' && (
            <div className="mt-4 w-64 mx-auto">
              <div className="h-1.5 w-full rounded-full bg-[#1a1a1a] overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-gray-500">{progress}% complete</p>
            </div>
          )}
        </div>

        <div className="absolute inset-0 -z-10 rounded-2xl bg-gradient-to-br from-emerald-500/0 via-emerald-500/0 to-emerald-500/5 opacity-0 transition-opacity group-hover:opacity-100" />
      </label>
    </div>
  );
}
