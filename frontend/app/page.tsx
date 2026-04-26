"use client";

import { useCallback, useState, useRef } from "react";
import { RefreshCw, Layers } from "lucide-react";
import UploadZone from "@/components/UploadZone";
import QuestionBar from "@/components/QuestionBar";
import InferenceOutput from "@/components/InferenceOutput";
import SceneGraphPanel from "@/components/SceneGraphPanel";
import { runInference, postCorrection, clearSession } from "@/lib/api";
import { InferenceResult } from "@/lib/types";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const sessionId = useRef<string>("");

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleQuestion = useCallback(
    async (question: string) => {
      if (!file) return;
      setLoading(true);
      setError(null);
      try {
        const res = await runInference(file, question, sessionId.current);
        sessionId.current = res.session_id;
        setResult(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    },
    [file]
  );

  const handleCorrect = useCallback(
    async (nodeId: string, correction: Record<string, unknown>) => {
      if (!sessionId.current) return;
      try {
        const res = await postCorrection({
          session_id: sessionId.current,
          node_id: nodeId,
          ...(correction as Record<string, never>),
        });
        if (res.success && res.graph && result) {
          setResult({ ...result, scene_graph: res.graph });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Correction failed");
      }
    },
    [result]
  );

  const handleReset = useCallback(async () => {
    if (sessionId.current) await clearSession(sessionId.current);
    sessionId.current = "";
    setFile(null);
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-violet-400" />
          <span className="font-semibold tracking-tight text-white/90">spatial-ml</span>
          <span className="text-white/20 text-sm hidden sm:inline">/ 3D construction reasoning</span>
        </div>
        {(file || result) && (
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
              Upload a construction site image
            </h1>
            <p className="text-sm text-white/40 mt-1">
              Get quantitative spatial estimates, object tracking, and safety alerts.
            </p>
          </div>

          <UploadZone onFile={handleFile} disabled={loading} />

          <QuestionBar
            onSubmit={handleQuestion}
            loading={loading}
            disabled={!file}
          />

          {error && (
            <div className="rounded-lg border border-red-500/20 bg-red-950/20 px-4 py-3 text-xs text-red-300">
              {error}
            </div>
          )}

          {!result && !loading && (
            <div className="text-xs text-white/20 leading-relaxed">
              Upload an image, then ask a spatial question — or use a suggestion above. The model will reason step-by-step and update the scene graph.
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
                <p className="text-xs text-white/30">Running spatial inference…</p>
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
