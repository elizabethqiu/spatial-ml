"use client";

import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Node,
  type Edge,
  type NodeProps,
  type Connection,
  Handle,
  Position,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { SceneGraph, SceneNode } from "@/lib/types";
import { X, Check, Trash2 } from "lucide-react";

// ---------------------------------------------------------------------------
// Custom node
// ---------------------------------------------------------------------------

type NodeData = {
  node: SceneNode;
  onSelect: (n: SceneNode) => void;
};

function ObjectNode({ data }: NodeProps) {
  const { node, onSelect } = data as NodeData;
  const dimmed = !node.in_frame;

  return (
    <>
      <Handle type="target" position={Position.Left} className="!bg-white/20 !border-white/10 !w-2 !h-2" />
      <div
        onClick={() => onSelect(node)}
        className={`
          px-3 py-2 rounded-lg border cursor-pointer select-none min-w-[110px]
          transition-all duration-150
          ${dimmed
            ? "border-white/10 bg-white/[0.03] text-white/30"
            : node.user_corrected
              ? "border-violet-500/50 bg-violet-950/40 text-white/90 shadow-[0_0_8px_rgba(139,92,246,0.2)]"
              : "border-white/15 bg-[#0f0f1a] text-white/80 hover:border-white/30"
          }
        `}
      >
        <div className="text-xs font-semibold truncate max-w-[120px]">{node.label}</div>
        <div className="flex items-center justify-between gap-2 mt-0.5">
          <span className={`text-[10px] ${dimmed ? "text-white/20" : "text-white/40"}`}>
            {node.depth_m.toFixed(1)} m
          </span>
          <span className={`text-[10px] tabular-nums ${dimmed ? "text-white/20" : "text-violet-400/70"}`}>
            {(node.confidence * 100).toFixed(0)}%
          </span>
        </div>
        {!node.in_frame && (
          <div className="text-[9px] text-white/25 mt-0.5 uppercase tracking-wider">off-screen</div>
        )}
        {node.user_corrected && (
          <div className="text-[9px] text-violet-400/80 mt-0.5 uppercase tracking-wider">corrected</div>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-white/20 !border-white/10 !w-2 !h-2" />
    </>
  );
}

const nodeTypes = { object: ObjectNode };

// ---------------------------------------------------------------------------
// Layout: simple left-to-right columns by depth bucket
// ---------------------------------------------------------------------------

function layoutNodes(nodes: SceneNode[]): { x: number; y: number }[] {
  if (nodes.length === 0) return [];

  const sorted = [...nodes].map((n, i) => ({ i, depth: n.depth_m }));
  sorted.sort((a, b) => a.depth - b.depth);

  // Bucket into ~3 depth columns
  const minD = sorted[0].depth;
  const maxD = sorted[sorted.length - 1].depth || minD + 1;
  const cols = 3;
  const colW = 200;
  const rowH = 90;

  const colCount = new Array(cols).fill(0);
  const positions: { x: number; y: number }[] = new Array(nodes.length);

  for (const { i, depth } of sorted) {
    const col = Math.min(cols - 1, Math.floor(((depth - minD) / (maxD - minD + 0.001)) * cols));
    const row = colCount[col]++;
    positions[i] = { x: col * colW + 40, y: row * rowH + 30 };
  }
  return positions;
}

// ---------------------------------------------------------------------------
// Correction drawer
// ---------------------------------------------------------------------------

interface DrawerProps {
  node: SceneNode;
  onClose: () => void;
  onCorrect: (nodeId: string, c: Record<string, unknown>) => void;
}

function CorrectionDrawer({ node, onClose, onCorrect }: DrawerProps) {
  const [label, setLabel] = useState(node.label);

  return (
    <div className="absolute bottom-0 left-0 right-0 z-10 bg-[#0d0d18] border-t border-white/10 p-4 rounded-b-lg">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-white/70">Edit node · <span className="font-mono text-violet-400">{node.id}</span></span>
        <button onClick={onClose} className="text-white/30 hover:text-white/60 transition-colors">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex gap-2 mb-2">
        <input
          className="flex-1 bg-white/5 border border-white/10 rounded px-2 py-1 text-xs outline-none focus:border-violet-500"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Label"
        />
        <button
          className="flex items-center gap-1 text-xs bg-violet-600 hover:bg-violet-500 px-2.5 py-1 rounded transition-colors"
          onClick={() => { onCorrect(node.id, { label }); onClose(); }}
        >
          <Check className="w-3 h-3" /> Save
        </button>
      </div>

      <div className="grid grid-cols-3 gap-1.5 text-[10px] text-white/40 mb-3">
        <div>depth: <span className="text-white/60">{node.depth_m.toFixed(2)} m</span></div>
        <div>conf: <span className="text-white/60">{(node.confidence * 100).toFixed(0)}%</span></div>
        <div>in-frame: <span className="text-white/60">{node.in_frame ? "yes" : "no"}</span></div>
        {node.position_3d && (
          <div className="col-span-3">
            pos: <span className="text-white/60 font-mono">
              ({node.position_3d.map((v) => v.toFixed(1)).join(", ")}) m
            </span>
          </div>
        )}
      </div>

      <button
        className="flex items-center gap-1 text-[11px] text-red-400/70 hover:text-red-400 transition-colors"
        onClick={() => { onCorrect(node.id, { delete: true }); onClose(); }}
      >
        <Trash2 className="w-3 h-3" /> Remove node
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

interface Props {
  graph: SceneGraph;
  onCorrect: (nodeId: string, correction: Record<string, unknown>) => void;
}

export default function SceneGraphPanel({ graph, onCorrect }: Props) {
  const [selectedNode, setSelectedNode] = useState<SceneNode | null>(null);

  const positions = useMemo(() => layoutNodes(graph.nodes), [graph.nodes]);

  const rfNodes: Node[] = useMemo(
    () =>
      graph.nodes.map((n, i) => ({
        id: n.id,
        type: "object",
        position: positions[i] ?? { x: 0, y: 0 },
        data: { node: n, onSelect: setSelectedNode },
        draggable: true,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.nodes, positions]
  );

  const rfEdges: Edge[] = useMemo(
    () =>
      graph.edges.map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        label: e.distance_m != null ? `${e.relation} · ${e.distance_m.toFixed(1)}m` : e.relation,
        labelStyle: { fill: "rgba(255,255,255,0.35)", fontSize: 9 },
        labelBgStyle: { fill: "#0d0d18", fillOpacity: 0.8 },
        style: {
          stroke:
            e.relation === "proximity"
              ? "rgba(139,92,246,0.5)"
              : e.relation === "above" || e.relation === "below"
                ? "rgba(99,179,237,0.5)"
                : "rgba(255,255,255,0.15)",
          strokeWidth: 1.5,
        },
        animated: e.relation === "proximity" && (e.distance_m ?? 999) < 2,
      })),
    [graph.edges]
  );

  const [nodes, , onNodesChange] = useNodesState(rfNodes);
  const [edges, , onEdgesChange] = useEdgesState(rfEdges);

  // Sync when graph prop changes
  const syncedNodes = rfNodes.length !== nodes.length ? rfNodes : nodes;
  const syncedEdges = rfEdges.length !== edges.length ? rfEdges : edges;

  if (graph.nodes.length === 0) {
    return (
      <div className="flex flex-col gap-2 h-full">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white/80">Scene Graph</h2>
          <span className="text-xs text-white/30">frame {graph.frame_count}</span>
        </div>
        <div className="flex-1 flex items-center justify-center text-xs text-white/25">
          No objects detected yet.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 h-full min-h-[380px]">
      <div className="flex items-center justify-between shrink-0">
        <h2 className="text-sm font-semibold text-white/80">Scene Graph</h2>
        <span className="text-xs text-white/30">
          {graph.nodes.length} nodes · {graph.edges.length} edges · frame {graph.frame_count}
        </span>
      </div>

      <div className="relative flex-1 rounded-lg border border-white/5 overflow-hidden bg-[#08080f]">
        <ReactFlow
          nodes={syncedNodes}
          edges={syncedEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.3}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={18}
            size={1}
            color="rgba(255,255,255,0.04)"
          />
          <Controls
            className="!bg-[#0d0d18] !border-white/10 !shadow-none"
            showInteractive={false}
          />
          <MiniMap
            nodeColor={(n) => {
              const sc = graph.nodes.find((gn) => gn.id === n.id);
              if (!sc?.in_frame) return "rgba(255,255,255,0.08)";
              if (sc.user_corrected) return "rgba(139,92,246,0.6)";
              return "rgba(255,255,255,0.25)";
            }}
            maskColor="rgba(8,8,15,0.7)"
            style={{ background: "#0d0d18", border: "1px solid rgba(255,255,255,0.06)" }}
          />
        </ReactFlow>

        {selectedNode && (
          <CorrectionDrawer
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
            onCorrect={onCorrect}
          />
        )}
      </div>
    </div>
  );
}
