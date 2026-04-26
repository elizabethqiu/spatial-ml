"""
Two-stage inference:
  1. SpaceLLaVA (remyxai/SpaceLLaVA) — vision grounding, produces spatial context
  2. K2-Think-V2 (LLM360/K2-Think-V2) — deep CoT reasoning over that context

The vision stage extracts spatial facts (objects, distances, depth).
The reasoning stage chains through those facts to produce structured output.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.scene_graph import SceneGraph


COT_SYSTEM = """\
You are a spatial reasoning assistant for construction site safety.
You receive a structured spatial description extracted from an image and must reason \
step-by-step about object relationships, distances, and safety concerns.

End your response with exactly one JSON block:
```json
{
  "estimates": [{"description": "...", "value_m": 1.23}],
  "safety_alerts": ["..."],
  "summary": "..."
}
```
"""


class SpatialInferenceEngine:
    def __init__(
        self,
        vision_model_path: str = "remyxai/SpaceLLaVA",
        reasoning_model_path: str | None = None,  # unused — K2 is API-only
        device: str | None = None,
        scene_graph: SceneGraph | None = None,
        load_reasoning: bool = True,  # kept for API compat; True = call K2 API
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_graph = scene_graph or SceneGraph()
        self.use_k2 = load_reasoning and bool(os.getenv("K2_API_KEY"))

        print(f"Loading vision model: {vision_model_path}")
        self.vision_processor = AutoProcessor.from_pretrained(
            vision_model_path, trust_remote_code=True
        )
        self.vision_model = AutoModelForCausalLM.from_pretrained(
            vision_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

        if self.use_k2:
            print("K2-Think-V2: using API at", os.getenv("K2_API_BASE", "https://api.k2think.ai/v1"))
        else:
            print("K2-Think-V2: disabled (set K2_API_KEY to enable)")

    # ------------------------------------------------------------------
    # Stage 1 — vision grounding (SpaceLLaVA, runs locally)
    # ------------------------------------------------------------------

    def _vision_pass(self, image: Image.Image, question: str) -> str:
        prompt = (
            "Describe the spatial layout of this construction site scene. "
            "For each visible object, estimate its distance from the camera in metres "
            "and its position relative to other objects. "
            f"User question: {question}"
        )
        inputs = self.vision_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.vision_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self.vision_processor.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Stage 2 — CoT reasoning (K2-Think-V2 via API)
    # ------------------------------------------------------------------

    def _reasoning_pass(self, spatial_context: str, question: str) -> dict:
        if not self.use_k2:
            return _parse_cot_output(spatial_context)

        api_base = os.getenv("K2_API_BASE", "https://api.k2think.ai/v1")
        api_key = os.getenv("K2_API_KEY", "")
        model = os.getenv("K2_MODEL", "MBZUAI-IFM/K2-Think-v2")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": (
                    f"Spatial context extracted from image:\n{spatial_context}\n\n"
                    f"Question: {question}"
                )},
            ],
            "stream": False,
        }

        resp = httpx.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_cot_output(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_frame(self, image: Image.Image, question: str | None = None) -> dict:
        timestamp = time.time()
        question = question or "Describe the spatial layout and any safety concerns."

        # Stage 1 — vision
        spatial_context = self._vision_pass(image, question)

        # Stage 2 — reasoning
        parsed = self._reasoning_pass(spatial_context, question)

        # Update scene graph with any detections (empty until expert annotation
        # is integrated into the inference path)
        alerts = self.scene_graph.update([], timestamp)
        parsed["safety_alerts"] = list(set(parsed.get("safety_alerts", []) + alerts))
        parsed["scene_graph"] = self.scene_graph.to_dict()
        parsed["cot_trace"] = spatial_context

        return parsed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_cot_output(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {"summary": text, "estimates": [], "safety_alerts": []}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--vision_model", default=os.getenv("VISION_MODEL_PATH", "remyxai/SpaceLLaVA"))
    parser.add_argument("--question", default=None)
    parser.add_argument("--no_reasoning", action="store_true", help="Skip K2 API reasoning stage")
    args = parser.parse_args()

    engine = SpatialInferenceEngine(
        vision_model_path=args.vision_model,
        load_reasoning=not args.no_reasoning,
    )
    image = Image.open(args.image).convert("RGB")
    result = engine.run_frame(image, args.question)
    print(json.dumps(result, indent=2))
