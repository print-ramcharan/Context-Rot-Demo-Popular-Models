import React from 'react';
import { AlertTriangle, Zap, BookOpen, Clock, Cpu } from 'lucide-react';

// Skeleton loading component
function SkeletonPane() {
  return (
    <div className="space-y-3 animate-in fade-in duration-300">
      <div className="h-4 w-3/4 rounded-lg bg-[#1a1a1a] animate-pulse" />
      <div className="h-4 w-full rounded-lg bg-[#1a1a1a] animate-pulse" style={{ animationDelay: '75ms' }} />
      <div className="h-4 w-5/6 rounded-lg bg-[#1a1a1a] animate-pulse" style={{ animationDelay: '150ms' }} />
      <div className="h-4 w-4/5 rounded-lg bg-[#1a1a1a] animate-pulse" style={{ animationDelay: '225ms' }} />
      <div className="h-4 w-2/3 rounded-lg bg-[#1a1a1a] animate-pulse" style={{ animationDelay: '300ms' }} />
      <div className="h-4 w-3/5 rounded-lg bg-[#1a1a1a] animate-pulse" style={{ animationDelay: '375ms' }} />
    </div>
  );
}

// Response pane component
function ResponsePane({ 
  type, 
  title, 
  icon: Icon, 
  accentColor, 
  response, 
  loading, 
  sources = [],
  showSources = false 
}) {
  const isStandard = type === 'standard';
  
  return (
    <div className={`
      group relative overflow-hidden rounded-2xl border-2 transition-all duration-300
      ${isStandard 
        ? 'border-red-500/20 bg-gradient-to-br from-red-500/5 to-red-600/5 hover:border-red-500/30' 
        : 'border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-emerald-600/5 hover:border-emerald-500/30'
      }
    `}>
      {/* Gradient overlay */}
      <div className={`absolute inset-0 bg-gradient-to-br opacity-0 transition-opacity group-hover:opacity-100 ${
        isStandard ? 'from-red-500/5 to-transparent' : 'from-emerald-500/5 to-transparent'
      }`} />

      <div className="relative space-y-6 p-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`rounded-xl border p-2.5 ${
              isStandard 
                ? 'border-red-500/30 bg-red-500/10' 
                : 'border-emerald-500/30 bg-emerald-500/10'
            }`}>
              <Icon className={`h-5 w-5 ${isStandard ? 'text-red-400' : 'text-emerald-400'}`} />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">{title}</h3>
              <p className="text-xs text-gray-500 mt-0.5">
                {isStandard ? 'No retrieved context' : 'With retrieved context'}
              </p>
            </div>
          </div>

          {/* Status badge */}
          {!loading && response && (
            <div className={`rounded-full px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${
              isStandard 
                ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
                : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            }`}>
              {isStandard ? 'Limited' : 'Enhanced'}
            </div>
          )}
        </div>

        {/* Response Text */}
        <div className="min-h-[200px] rounded-xl border border-[#1a1a1a] bg-[#0b0b0b] p-5">
          {loading ? (
            <SkeletonPane />
          ) : response?.text ? (
            <p className="text-sm leading-relaxed text-gray-300 whitespace-pre-wrap">
              {response.text}
            </p>
          ) : (
            <p className="text-sm text-gray-500 italic">
              No response yet. Upload a document and ask a question.
            </p>
          )}
        </div>

        {/* Context Section (RAG only) */}
        {!isStandard && response?.context_used && (
          <details className="group/details overflow-hidden rounded-xl border border-emerald-500/20 bg-emerald-500/5 transition-all">
            <summary className="cursor-pointer px-4 py-3 text-xs font-semibold text-emerald-400 hover:bg-emerald-500/10 transition-colors flex items-center justify-between">
              <span className="flex items-center gap-2">
                <BookOpen className="h-3.5 w-3.5" />
                View Retrieved Context
              </span>
              <span className="text-gray-500 group-open/details:rotate-180 transition-transform">▼</span>
            </summary>
            <div className="border-t border-emerald-500/20 bg-[#0b0b0b] p-4">
              <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono">
                {response.context_used}
              </pre>
            </div>
          </details>
        )}

        {/* Source Citations (RAG only) */}
        {showSources && sources.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
              <div className="h-px flex-1 bg-emerald-500/20" />
              <span className="flex items-center gap-2">
                <BookOpen className="h-3.5 w-3.5" />
                Source Citations
              </span>
              <div className="h-px flex-1 bg-emerald-500/20" />
            </div>

            <ul className="space-y-2">
              {sources.map((source, i) => (
                <li 
                  key={i}
                  className="flex items-start gap-3 rounded-lg border border-[#1a1a1a] bg-[#0b0b0b] p-3 text-xs transition-colors hover:border-emerald-500/20 hover:bg-emerald-500/5"
                >
                  <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-md bg-emerald-500/10 text-[10px] font-bold text-emerald-400">
                    {i + 1}
                  </span>
                  <span className="text-gray-400 leading-relaxed">
                    {source}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metadata */}
        {response && (
          <div className="flex flex-wrap items-center gap-4 border-t border-[#1a1a1a] pt-4 text-[11px]">
            <div className="flex items-center gap-1.5 text-gray-500">
              <Cpu className="h-3.5 w-3.5" />
              <span className="font-mono">{response.model || 'gemini'}</span>
            </div>
            <div className="flex items-center gap-1.5 text-gray-500">
              <Clock className="h-3.5 w-3.5" />
              <span className="font-mono">{response.latency_ms?.toFixed?.(0) || 0}ms</span>
            </div>
            <div className="flex items-center gap-1.5 text-gray-500">
              <span className="font-mono">
                {response.tokens_used?.total || 0} tokens
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ComparisonView({ standard, rag, loading, sources = [] }) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Standard LLM Response */}
      <ResponsePane
        type="standard"
        title="Context Rot (Standard)"
        icon={AlertTriangle}
        accentColor="red"
        response={standard}
        loading={loading}
      />

      {/* RAG Response */}
      <ResponsePane
        type="rag"
        title="RAG Optimized"
        icon={Zap}
        accentColor="emerald"
        response={rag}
        loading={loading}
        sources={sources}
        showSources={true}
      />
    </div>
  );
}