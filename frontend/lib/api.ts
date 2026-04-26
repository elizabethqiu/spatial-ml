import { InferenceResult, SceneGraph } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function runInference(
  file: File,
  question: string,
  sessionId: string
): Promise<InferenceResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("question", question);
  form.append("session_id", sessionId);

  const res = await fetch(`${BASE}/infer`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Inference failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchGraph(sessionId: string): Promise<SceneGraph> {
  const res = await fetch(`${BASE}/graph/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch scene graph");
  return res.json();
}

export async function postCorrection(params: {
  session_id: string;
  node_id: string;
  label?: string;
  position_3d?: [number, number, number];
  depth_m?: number;
  delete?: boolean;
}): Promise<{ success: boolean; graph: SceneGraph | null; message: string }> {
  const res = await fetch(`${BASE}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Correction failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/graph/${sessionId}`, { method: "DELETE" });
}
