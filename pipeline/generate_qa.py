"""
Generate quantitative + qualitative VQA pairs from annotated frames
using Claude. Runs with a thread pool for parallel API calls.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import anthropic
import numpy as np


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


SYSTEM_PROMPT = """\
You are a spatial reasoning expert for construction site safety.
Given a scene description and object list with 3D positions, generate diverse VQA pairs.
Produce a JSON array where each element has:
  "question": string
  "answer": string
  "type": one of ["quantitative_distance", "binary_predicate", "relative_position", "safety_alert"]
  "objects_referenced": list of object labels from the scene

Rules:
- Quantitative answers must include units (metres).
- Binary predicate answers are exactly "yes" or "no".
- Safety alerts should flag any proximity < 2 m between a worker and heavy equipment.
- Generate at least 2 examples of each type if the scene supports it.
"""


def build_user_prompt(caption: str, masks: list[dict], depth_stats: dict) -> str:
    objects = [f"- {m['label']} (confidence {m['score']:.2f})" for m in masks]
    obj_str = "\n".join(objects) if objects else "No objects detected."
    return f"""Scene caption:
{caption}

Detected objects:
{obj_str}

Depth statistics (metres):
  mean: {depth_stats['mean']:.2f}
  min:  {depth_stats['min']:.2f}
  max:  {depth_stats['max']:.2f}

Generate VQA pairs for this scene."""


def generate_qa(caption: str, masks: list[dict], depth: np.ndarray) -> list[dict]:
    depth_stats = {
        "mean": float(np.mean(depth)),
        "min": float(np.min(depth)),
        "max": float(np.max(depth)),
    }
    client = get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(caption, masks, depth_stats)}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def process_one(caption_path: Path, out_dir: Path, frames_dir: Path) -> str:
    ann_dir = caption_path.parent
    depth_path = ann_dir / "depth.npy"
    masks_path = ann_dir / "masks.json"
    out_path = out_dir / f"{ann_dir.name}.json"

    if out_path.exists():
        return f"skip {ann_dir.name}"
    if not depth_path.exists() or not masks_path.exists():
        return f"missing {ann_dir.name}"

    caption = caption_path.read_text().strip()
    masks = json.loads(masks_path.read_text())
    depth = np.load(depth_path)

    image_path = next(
        (str(p) for p in frames_dir.rglob(f"{ann_dir.name}.*")
         if p.suffix in {".jpg", ".png"}),
        "",
    )

    qa_pairs = generate_qa(caption, masks, depth)
    for pair in qa_pairs:
        pair["image_path"] = image_path
        pair.pop("__IMAGE_PATH__", None)
    out_path.write_text(json.dumps(qa_pairs, indent=2))
    return f"ok {ann_dir.name} ({len(qa_pairs)} pairs)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/vqa_pairs"))
    parser.add_argument("--frames_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ann_dirs = sorted(args.annotations_dir.rglob("caption.txt"))
    if args.limit:
        ann_dirs = ann_dirs[: args.limit]

    total = len(ann_dirs)
    done = 0
    errors = 0
    lock = Lock()

    print(f"Processing {total} frames with {args.workers} workers…")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, p, args.out_dir, args.frames_dir): p
            for p in ann_dirs
        }
        for fut in as_completed(futures):
            with lock:
                done += 1
                try:
                    msg = fut.result()
                    status = "·" if msg.startswith("skip") else "✓" if msg.startswith("ok") else "?"
                    print(f"[{done}/{total}] {status} {msg}")
                except Exception as e:
                    errors += 1
                    print(f"[{done}/{total}] ERROR {futures[fut].parent.name}: {e}")

    print(f"\nDone: {done - errors} ok, {errors} errors")


if __name__ == "__main__":
    main()
