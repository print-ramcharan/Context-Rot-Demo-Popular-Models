import React from 'react';
import { Cpu, Layers3 } from 'lucide-react';

export default function Header() {
  return (
    <header className="w-full border-b border-[#1a1a1a] bg-[#050505]">
      <div className="mx-auto max-w-6xl px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-[#1a1a1a] bg-[#0b0b0b] p-2">
            <Layers3 className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">Context Engine</h1>
            <p className="text-xs text-gray-400">Modular RAG Architecture · 2026 Edition</p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs text-gray-400">
          <Cpu className="h-4 w-4 text-gray-500" />
          <span>Gemini · FAISS · MiniLM</span>
        </div>
      </div>
    </header>
  );
}