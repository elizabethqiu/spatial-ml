export interface SceneNode {
  id: string;
  label: string;
  bbox_xyxy: [number, number, number, number];
  depth_m: number;
  position_3d: [number, number, number];
  confidence: number;
  last_seen: number;
  in_frame: boolean;
  user_corrected: boolean;
}

export interface SceneEdge {
  source: string;
  target: string;
  relation: string;
  distance_m: number | null;
  confidence: number;
}

export interface SceneGraph {
  timestamp: number;
  frame_count: number;
  nodes: SceneNode[];
  edges: SceneEdge[];
}

export interface DistanceEstimate {
  description: string;
  value_m: number;
}

export interface InferenceResult {
  session_id: string;
  summary: string;
  cot_trace: string;
  estimates: DistanceEstimate[];
  safety_alerts: string[];
  scene_graph: SceneGraph;
}
