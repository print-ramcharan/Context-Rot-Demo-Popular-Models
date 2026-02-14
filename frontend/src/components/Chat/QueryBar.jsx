import React from 'react';
import { Sparkles } from 'lucide-react';

export default function QueryBar({ value, onChange, onSubmit, loading }) {
  return (
    <form
      onSubmit={onSubmit}
      className="mx-auto flex w-full max-w-3xl items-center gap-3 rounded-2xl border border-[#1a1a1a] bg-[#0b0b0b] px-4 py-3"
    >
      <Sparkles className="h-5 w-5 text-emerald-400" />
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Ask a question about your indexed sources..."
        className="w-full bg-transparent text-sm text-white placeholder:text-gray-500 outline-none"
      />
      <button
        type="submit"
        disabled={loading}
        className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-xs font-semibold text-emerald-300 transition hover:bg-emerald-500/20 disabled:opacity-50"
      >
        {loading ? 'Thinking...' : 'Query'}
      </button>
    </form>
  );
}