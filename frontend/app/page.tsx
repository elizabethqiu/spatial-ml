"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw, Layers } from "lucide-react";
import InferenceOutput from "@/components/InferenceOutput";
import SceneGraphPanel from "@/components/SceneGraphPanel";
import { InferenceResult } from "@/lib/types";

type ExampleIndex = {
  examples: Array<{
    id: string;
    title: string;
    imageUrl: string;
    question: string;
    resultPath: string;
  }>;
};

export default function Home() {
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [index, setIndex] = useState<ExampleIndex | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("/examples/index.json");
        if (!res.ok) throw new Error(`Failed to load examples: ${res.status}`);
        const data = (await res.json()) as ExampleIndex;
        if (cancelled) return;
        setIndex(data);
        setSelectedId((prev) => prev || data.examples[0]?.id || "");
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load examples");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = useMemo(
    () => index?.examples.find((e) => e.id === selectedId) ?? null,
    [index, selectedId]
  );

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(selected.resultPath);
        if (!res.ok) throw new Error(`Failed to load result: ${res.status}`);
        const data = (await res.json()) as InferenceResult;
        if (cancelled) return;
        setResult(data);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load result");
        setResult(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const handleReset = useCallback(() => {
    setSelectedId(index?.examples[0]?.id ?? "");
    setError(null);
  }, [index]);

  const handleCorrect = useCallback(() => {
    // Demo is fully static: no backend correction loop.
    // (We intentionally no-op to keep the UI simple and offline-friendly.)
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-violet-400" />
          <span className="font-semibold tracking-tight text-white/90">Spatial Stack</span>
          <span className="text-white/20 text-sm hidden sm:inline">/ HackTech 2026</span>
        </div>
        {(selectedId || result) && (
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-white/30 hover:text-white/60 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reset
          </button>
        )}
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col lg:flex-row gap-0 overflow-hidden">
        {/* Left column */}
        <div className="flex flex-col gap-4 p-6 lg:w-[420px] lg:border-r border-white/5 lg:overflow-y-auto">
          <div>
            <h1 className="text-lg font-semibold text-white/90 leading-snug">
              Pick an example
            </h1>
            <p className="text-sm text-white/40 mt-1">
              Demo-only viewer (no backend required).
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-[11px] uppercase tracking-widest text-white/30">
              Example
            </label>
            <select
              className="bg-white/[0.04] border border-white/10 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-violet-500/60 transition-colors disabled:opacity-40"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              disabled={!index || loading}
            >
              {(index?.examples ?? []).map((ex) => (
                <option key={ex.id} value={ex.id}>
                  {ex.title}
                </option>
              ))}
            </select>
            {selected?.question && (
              <div className="text-xs text-white/35 leading-relaxed">
                <span className="text-white/25">Question:</span>{" "}
                <span className="text-white/55">{selected.question}</span>
              </div>
            )}
          </div>

          {selected?.imageUrl && (
            <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selected.imageUrl}
                alt={selected.title}
                className="w-full h-full object-contain max-h-72"
              />
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-500/20 bg-red-950/20 px-4 py-3 text-xs text-red-300">
              {error}
            </div>
          )}

          {!result && !loading && (
            <div className="text-xs text-white/20 leading-relaxed">
              Select an example to view its precomputed output.
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="flex-1 flex flex-col lg:flex-row gap-0 overflow-hidden">
          {/* Inference output */}
          <div className="flex-1 p-6 overflow-y-auto border-b lg:border-b-0 lg:border-r border-white/5">
            {loading && (
              <div className="flex flex-col items-center justify-center h-40 gap-3">
                <span className="w-6 h-6 rounded-full border-2 border-white/20 border-t-violet-400 animate-spin" />
                <p className="text-xs text-white/30">Loading example…</p>
              </div>
            )}
            {!loading && result && <InferenceOutput result={result} />}
            {!loading && !result && (
              <div className="flex items-center justify-center h-full text-white/15 text-sm">
                Results will appear here
              </div>
            )}
          </div>

          {/* Scene graph */}
          <div className="p-6 lg:w-72 xl:w-80 overflow-y-auto">
            {result ? (
              <SceneGraphPanel
                graph={result.scene_graph}
                onCorrect={handleCorrect}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-white/15 text-sm">
                Scene graph will appear here
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
