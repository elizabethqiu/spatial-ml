"""
User correction loop: accepts corrections to the scene graph,
applies them immediately, and logs them as future training signal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from model.scene_graph import SceneGraph


CORRECTION_LOG = Path("data/corrections.jsonl")


def apply_and_log(
    graph: SceneGraph,
    node_id: str,
    correction: dict,
    session_id: str = "",
) -> dict:
    """
    Apply a correction to a node and write it to the correction log.

    correction keys (all optional):
      label       — rename the object
      position_3d — override 3D position
      depth_m     — override metric depth
      delete      — bool, remove the node entirely
    """
    success = graph.apply_correction(node_id, correction)
    entry = {
        "timestamp": time.time(),
        "session_id": session_id,
        "node_id": node_id,
        "correction": correction,
        "applied": success,
        "frame_count": graph.frame_count,
    }
    CORRECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTION_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "success": success,
        "graph": graph.to_dict() if success else None,
        "message": "Correction applied." if success else f"Node {node_id} not found.",
    }


def load_correction_log() -> list[dict]:
    if not CORRECTION_LOG.exists():
        return []
    return [json.loads(line) for line in CORRECTION_LOG.read_text().splitlines() if line.strip()]


def export_corrections_as_training_pairs(out_path: Path):
    """
    Convert logged corrections into (before, after) training pairs
    that can be used for reinforcement / fine-tuning signal.
    """
    entries = load_correction_log()
    pairs = []
    for e in entries:
        if not e["applied"]:
            continue
        c = e["correction"]
        pair = {
            "node_id": e["node_id"],
            "session_id": e["session_id"],
            "timestamp": e["timestamp"],
        }
        if "label" in c:
            pair["type"] = "relabel"
            pair["corrected_label"] = c["label"]
        elif "position_3d" in c:
            pair["type"] = "reposition"
            pair["corrected_position"] = c["position_3d"]
        elif c.get("delete"):
            pair["type"] = "deletion"
        else:
            pair["type"] = "attribute_update"
            pair["update"] = c
        pairs.append(pair)

    out_path.write_text(json.dumps(pairs, indent=2))
    print(f"Exported {len(pairs)} correction pairs to {out_path}")
    return pairs
