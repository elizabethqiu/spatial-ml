"use client";

import { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Ruler } from "lucide-react";
import { InferenceResult } from "@/lib/types";

interface Props {
  result: InferenceResult;
}

export default function InferenceOutput({ result }: Props) {
  const [cotOpen, setCotOpen] = useState(false);

  return (
    <div className="flex flex-col gap-4">
      {/* Safety alerts */}
      {result.safety_alerts.length > 0 && (
        <div className="rounded-lg border border-red-500/30 bg-red-950/20 p-3 space-y-1.5">
          <div className="flex items-center gap-2 text-red-400 text-sm font-medium">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            Safety Alerts
          </div>
          {result.safety_alerts.map((a, i) => (
            <p key={i} className="text-xs text-red-300/80 pl-6">{a}</p>
          ))}
        </div>
      )}

      {/* Summary */}
      <div className="rounded-lg bg-white/[0.03] border border-white/5 p-4">
        <p className="text-sm text-white/80 leading-relaxed">{result.summary}</p>
      </div>

      {/* Distance estimates */}
      {result.estimates.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] uppercase tracking-widest text-white/30">Estimates</p>
          {result.estimates.map((e, i) => (
            <div key={i} className="flex items-center gap-3 rounded-lg bg-white/[0.02] border border-white/5 px-3 py-2">
              <Ruler className="w-3.5 h-3.5 text-violet-400 shrink-0" />
              <span className="flex-1 text-xs text-white/60">{e.description}</span>
              <span className="text-sm font-mono font-medium text-violet-300">{e.value_m.toFixed(2)} m</span>
            </div>
          ))}
        </div>
      )}

      {/* CoT trace (collapsible) */}
      <div className="rounded-lg border border-white/5 overflow-hidden">
        <button
          className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-white/40 hover:text-white/60 hover:bg-white/[0.02] transition-colors"
          onClick={() => setCotOpen((o) => !o)}
        >
          {cotOpen ? <ChevronDown className="w-3.5 h-3.5 shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 shrink-0" />}
          Reasoning trace
        </button>
        {cotOpen && (
          <pre className="px-4 py-3 text-[11px] text-white/40 font-mono whitespace-pre-wrap leading-relaxed bg-white/[0.015] max-h-64 overflow-y-auto">
            {result.cot_trace}
          </pre>
        )}
      </div>
    </div>
  );
}
