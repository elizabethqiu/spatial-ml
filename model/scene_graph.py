"""
Persistent temporal scene graph for construction site objects.

Nodes  — detected objects with 3D position, depth, bounding box, confidence.
Edges  — spatial relationships (distance, proximity, occlusion, above/below).

Objects that leave frame are retained with a last_seen timestamp and
re-identified on return via IoU + embedding similarity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    label: str
    bbox_xyxy: list[float]          # image-space [x1,y1,x2,y2]
    depth_m: float                  # metric depth estimate
    position_3d: list[float]        # [X, Y, Z] in gravity-aligned world coords
    confidence: float
    embedding: list[float] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    in_frame: bool = True
    user_corrected: bool = False
    corrections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "bbox_xyxy": self.bbox_xyxy,
            "depth_m": self.depth_m,
            "position_3d": self.position_3d,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "in_frame": self.in_frame,
            "user_corrected": self.user_corrected,
        }


@dataclass
class Edge:
    source_id: str
    target_id: str
    relation: str           # "proximity" | "above" | "below" | "occludes" | "adjacent"
    distance_m: float | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "distance_m": self.distance_m,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

PROXIMITY_THRESHOLD_M = 5.0   # metres — closer than this → proximity edge
SAFETY_THRESHOLD_M = 2.0      # metres — worker + equipment → safety alert
REIDENTIFY_IOU_MIN = 0.3
REIDENTIFY_EMB_SIM_MIN = 0.7
FORGET_AFTER_S = 300.0        # seconds before a missing object is dropped


class SceneGraph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.timestamp: float = 0.0
        self.frame_count: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, detections: list[dict], timestamp: float | None = None) -> list[str]:
        """
        Ingest a new frame's detections. Returns list of safety alert messages.

        Each detection dict:
          label, bbox_xyxy, depth_m, position_3d, confidence, embedding (optional)
        """
        self.timestamp = timestamp or time.time()
        self.frame_count += 1

        matched_ids = set()
        for det in detections:
            node_id = self._match_or_create(det)
            matched_ids.add(node_id)

        # Mark unmatched in-frame nodes as out-of-frame
        for nid, node in self.nodes.items():
            if nid not in matched_ids and node.in_frame:
                node.in_frame = False

        self._prune_old_nodes()
        self._rebuild_edges()
        return self._safety_alerts()

    def _match_or_create(self, det: dict) -> str:
        best_id, best_score = None, 0.0
        for nid, node in self.nodes.items():
            if node.label != det["label"]:
                continue
            iou = _bbox_iou(node.bbox_xyxy, det["bbox_xyxy"])
            emb_sim = _cosine_sim(node.embedding, det.get("embedding", [])) if node.embedding else 0.0
            score = 0.6 * iou + 0.4 * emb_sim
            if score > best_score:
                best_score, best_id = score, nid

        if best_id and best_score >= REIDENTIFY_IOU_MIN:
            node = self.nodes[best_id]
            node.bbox_xyxy = det["bbox_xyxy"]
            node.depth_m = det["depth_m"]
            node.position_3d = det["position_3d"]
            node.confidence = det["confidence"]
            node.embedding = det.get("embedding", node.embedding)
            node.last_seen = self.timestamp
            node.in_frame = True
            return best_id

        node = Node(
            id=str(uuid.uuid4())[:8],
            label=det["label"],
            bbox_xyxy=det["bbox_xyxy"],
            depth_m=det["depth_m"],
            position_3d=det["position_3d"],
            confidence=det["confidence"],
            embedding=det.get("embedding", []),
            last_seen=self.timestamp,
        )
        self.nodes[node.id] = node
        return node.id

    def _prune_old_nodes(self):
        to_drop = [
            nid for nid, n in self.nodes.items()
            if not n.in_frame and (self.timestamp - n.last_seen) > FORGET_AFTER_S
        ]
        for nid in to_drop:
            del self.nodes[nid]

    def _rebuild_edges(self):
        self.edges = []
        node_list = list(self.nodes.values())
        for i, a in enumerate(node_list):
            for b in node_list[i + 1:]:
                if not a.position_3d or not b.position_3d:
                    continue
                dist = _euclidean(a.position_3d, b.position_3d)
                if dist <= PROXIMITY_THRESHOLD_M:
                    self.edges.append(Edge(a.id, b.id, "proximity", dist))
                dz = b.position_3d[2] - a.position_3d[2]
                if abs(dz) > 1.0:
                    rel = "above" if dz > 0 else "below"
                    self.edges.append(Edge(a.id, b.id, rel, dist))

    def _safety_alerts(self) -> list[str]:
        alerts = []
        worker_ids = {n.id for n in self.nodes.values() if n.label == "worker" and n.in_frame}
        equipment_labels = {"crane", "excavator", "concrete pump", "forklift", "bulldozer"}
        equipment_ids = {
            n.id for n in self.nodes.values()
            if n.label in equipment_labels and n.in_frame
        }
        for e in self.edges:
            pair = {e.source_id, e.target_id}
            if pair & worker_ids and pair & equipment_ids:
                if e.distance_m is not None and e.distance_m < SAFETY_THRESHOLD_M:
                    w = self.nodes.get((pair & worker_ids).pop())
                    eq = self.nodes.get((pair & equipment_ids).pop())
                    if w and eq:
                        alerts.append(
                            f"SAFETY: worker within {e.distance_m:.1f}m of {eq.label}"
                        )
        return alerts

    # ------------------------------------------------------------------
    # User corrections
    # ------------------------------------------------------------------

    def apply_correction(self, node_id: str, correction: dict) -> bool:
        """
        correction may contain: label, position_3d, depth_m, or delete=True.
        Returns True if the node was found and updated.
        """
        node = self.nodes.get(node_id)
        if not node:
            return False

        if correction.get("delete"):
            del self.nodes[node_id]
            self._rebuild_edges()
            return True

        for key in ("label", "position_3d", "depth_m"):
            if key in correction:
                setattr(node, key, correction[key])

        node.user_corrected = True
        node.corrections.append({**correction, "timestamp": self.timestamp})
        self._rebuild_edges()
        return True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_count": self.frame_count,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        g = cls()
        g.timestamp = data["timestamp"]
        g.frame_count = data["frame_count"]
        for nd in data["nodes"]:
            node = Node(**{k: nd[k] for k in Node.__dataclass_fields__ if k in nd})
            g.nodes[node.id] = node
        for ed in data["edges"]:
            g.edges.append(Edge(**{k: ed[k] for k in Edge.__dataclass_fields__ if k in ed}))
        return g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0 else 0.0


def _euclidean(a: list[float], b: list[float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)
