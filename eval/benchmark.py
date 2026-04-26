"""
Evaluate the finetuned model on four metrics:

1. Binary spatial predicate accuracy  (yes/no questions)
2. Quantitative distance MAE (mean absolute error in metres)
3. Object re-identification rate across occlusion
4. Scene graph consistency across frames

Usage:
  python eval/benchmark.py --model checkpoints/spatialvlm-construction \
                            --eval_dir data/eval \
                            --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.scene_graph import SceneGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, processor


def predict(model, processor, image: Image.Image, question: str, device: str, max_tokens: int = 128) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.decode(ids[0], skip_special_tokens=True).strip()


def extract_metres(text: str) -> float | None:
    m = re.search(r"([\d.]+)\s*m(?:etre|eter)?s?", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Metric 1: Binary predicate accuracy
# ---------------------------------------------------------------------------

def eval_binary(model, processor, samples: list[dict], device: str) -> dict:
    correct = 0
    for s in samples:
        image = Image.open(s["image_path"]).convert("RGB")
        pred = predict(model, processor, image, s["question"], device).lower()
        gt = s["answer"].lower()
        if ("yes" in pred) == ("yes" in gt):
            correct += 1
    acc = correct / len(samples) if samples else 0.0
    return {"binary_accuracy": round(acc, 4), "n": len(samples)}


# ---------------------------------------------------------------------------
# Metric 2: Quantitative distance MAE
# ---------------------------------------------------------------------------

def eval_quantitative(model, processor, samples: list[dict], device: str) -> dict:
    errors = []
    skipped = 0
    for s in samples:
        image = Image.open(s["image_path"]).convert("RGB")
        pred_text = predict(model, processor, image, s["question"], device)
        pred_m = extract_metres(pred_text)
        gt_m = extract_metres(s["answer"])
        if pred_m is None or gt_m is None:
            skipped += 1
            continue
        errors.append(abs(pred_m - gt_m))
    mae = float(np.mean(errors)) if errors else float("nan")
    return {"distance_mae_m": round(mae, 3), "n": len(errors), "skipped": skipped}


# ---------------------------------------------------------------------------
# Metric 3: Re-identification rate
# ---------------------------------------------------------------------------

def eval_reid(sequences: list[list[dict]]) -> dict:
    """
    sequences: each is a list of frame dicts with "detections" and "ground_truth_ids".
    We run the scene graph over each sequence and count how often a leaving object
    is correctly re-identified on return.
    """
    total, correct = 0, 0
    for seq in sequences:
        graph = SceneGraph()
        id_map: dict[str, str] = {}  # gt_id → graph node_id
        for frame in seq:
            graph.update(frame["detections"], frame.get("timestamp"))
            for gt_id, det in zip(frame.get("ground_truth_ids", []), frame["detections"]):
                graph_id = _find_matching_node(graph, det)
                if graph_id:
                    prev = id_map.get(gt_id)
                    if prev is not None:
                        total += 1
                        if prev == graph_id:
                            correct += 1
                    id_map[gt_id] = graph_id
    rate = correct / total if total else float("nan")
    return {"reid_rate": round(rate, 4), "total_events": total}


def _find_matching_node(graph: SceneGraph, det: dict) -> str | None:
    from model.scene_graph import _bbox_iou
    best_id, best_iou = None, 0.0
    for nid, node in graph.nodes.items():
        if node.label == det["label"]:
            iou = _bbox_iou(node.bbox_xyxy, det["bbox_xyxy"])
            if iou > best_iou:
                best_iou, best_id = iou, nid
    return best_id if best_iou > 0.3 else None


# ---------------------------------------------------------------------------
# Metric 4: Scene graph consistency
# ---------------------------------------------------------------------------

def eval_graph_consistency(sequences: list[list[dict]]) -> dict:
    """
    Measure how consistent edge sets are across adjacent frames of the same sequence.
    A stable scene should have high Jaccard similarity between consecutive edge sets.
    """
    sims = []
    for seq in sequences:
        graph = SceneGraph()
        prev_edges: set | None = None
        for frame in seq:
            graph.update(frame["detections"])
            curr_edges = {
                (e.source_id, e.target_id, e.relation)
                for e in graph.edges
            }
            if prev_edges is not None:
                inter = len(curr_edges & prev_edges)
                union = len(curr_edges | prev_edges)
                sims.append(inter / union if union else 1.0)
            prev_edges = curr_edges
    mean_sim = float(np.mean(sims)) if sims else float("nan")
    return {"graph_consistency_jaccard": round(mean_sim, 4), "n_transitions": len(sims)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval_dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = load_model(args.model, args.device)

    def load_split(name: str) -> list[dict]:
        p = args.eval_dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else []

    binary_samples = load_split("binary")
    quant_samples = load_split("quantitative")
    reid_seqs = load_split("reid_sequences")
    consistency_seqs = load_split("consistency_sequences")

    results = {}
    if binary_samples:
        results["binary"] = eval_binary(model, processor, binary_samples, args.device)
    if quant_samples:
        results["quantitative"] = eval_quantitative(model, processor, quant_samples, args.device)
    if reid_seqs:
        results["reid"] = eval_reid(reid_seqs)
    if consistency_seqs:
        results["consistency"] = eval_graph_consistency(consistency_seqs)

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
