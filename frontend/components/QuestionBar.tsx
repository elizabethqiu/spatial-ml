"use client";

import { FormEvent, useState } from "react";
import { Send } from "lucide-react";

interface Props {
  onSubmit: (question: string) => void;
  loading: boolean;
  disabled?: boolean;
}

const SUGGESTIONS = [
  "How far is the worker from the trench edge?",
  "Is any worker within 2 metres of heavy equipment?",
  "What objects are visible and where are they?",
  "Describe relative positions of all detected objects.",
];

export default function QuestionBar({ onSubmit, loading, disabled }: Props) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const q = value.trim();
    if (!q || loading || disabled) return;
    onSubmit(q);
    setValue("");
  };

  return (
    <div className="flex flex-col gap-2">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          className="flex-1 bg-white/[0.04] border border-white/10 rounded-lg px-4 py-2.5 text-sm outline-none placeholder:text-white/25 focus:border-violet-500/60 transition-colors disabled:opacity-40"
          placeholder="Ask a spatial question…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled || loading}
        />
        <button
          type="submit"
          disabled={!value.trim() || loading || disabled}
          className="flex items-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
        >
          {loading ? (
            <span className="w-4 h-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
          {loading ? "Analyzing…" : "Ask"}
        </button>
      </form>

      <div className="flex flex-wrap gap-1.5">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            className="text-[11px] text-white/30 bg-white/[0.03] hover:bg-white/[0.06] hover:text-white/50 border border-white/5 rounded px-2 py-0.5 transition-colors disabled:opacity-30"
            onClick={() => { setValue(s); }}
            disabled={disabled || loading}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
