"""
Expert annotation pipeline: depth estimation + open-vocabulary segmentation.

For each extracted frame, produces three files in data/annotations/<frame_stem>/:
    depth.npy        — float32 H×W metric depth array (metres)
    masks.json       — list of {"label": str, "score": float, "bbox_xyxy": [...]}
    caption.txt      — single-line scene caption

Primary path: VQASynth DepthEstimator (Apple DepthPro via ONNX) + Localizer
              (Florence-2 + SAM2) — requires Linux + CUDA.
Fallback path: transformers Depth-Anything-V2 + DETR object detection
              — runs on macOS / CPU for local iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Depth conversion helpers
# ---------------------------------------------------------------------------

def depth_pil_to_numpy(depth_pil: Image.Image) -> np.ndarray:
    arr = np.array(depth_pil).astype(np.float32)
    return arr / 1000.0  # mm → metres


# ---------------------------------------------------------------------------
# Fallback annotator (transformers only, no VQASynth)
# ---------------------------------------------------------------------------

class _FallbackAnnotator:
    """
    Uses Depth-Anything-V2 for relative depth and DETR for object detection.
    Depths are relative (not metric), but give the pipeline something real to
    work with during local dev / Mac iteration.
    """

    def __init__(self):
        from transformers import pipeline as hf_pipeline
        print("  Loading Depth-Anything-V2 (CPU fallback)…")
        self._depth_pipe = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        print("  Loading DETR object detector (CPU fallback)…")
        self._det_pipe = hf_pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            threshold=0.5,
        )

    def depth(self, image: Image.Image) -> np.ndarray:
        result = self._depth_pipe(image)
        arr = np.array(result["depth"]).astype(np.float32)
        # Depth-Anything returns relative depth; normalise to [0.5, 50] m heuristic
        arr = arr / arr.max() * 50.0
        return arr

    def detect(self, image: Image.Image) -> tuple[list[np.ndarray], list, list[str]]:
        results = self._det_pipe(image)
        masks, bboxes, captions = [], [], []
        W, H = image.size
        for r in results:
            box = r["box"]
            bbox = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
            bboxes.append(bbox)
            captions.append(r["label"])
            # No pixel mask from DETR — return empty array placeholder
            masks.append(np.zeros((H, W), dtype=np.uint8))
        return masks, bboxes, captions


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate_frame(
    image_path: Path,
    out_dir: Path,
    depth_estimator,
    localizer,
    fallback: bool = False,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_path = out_dir / "depth.npy"
    masks_path = out_dir / "masks.json"
    caption_path = out_dir / "caption.txt"

    if depth_path.exists() and masks_path.exists() and caption_path.exists():
        return True

    try:
        image = Image.open(image_path).convert("RGB")

        if fallback:
            depth = depth_estimator.depth(image)
            masks_uint8, bboxes_or_points, captions = depth_estimator.detect(image)
        else:
            depth_pil, _ = depth_estimator.run(image)
            depth = depth_pil_to_numpy(depth_pil)
            masks_uint8, bboxes_or_points, captions = localizer.run(image)

        np.save(depth_path, depth)

        mask_records = []
        for cap, bbox in zip(captions, bboxes_or_points):
            record: dict = {"label": cap, "score": 1.0}
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                record["bbox_xyxy"] = [float(v) for v in bbox]
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 2:
                x, y = float(bbox[0]), float(bbox[1])
                record["bbox_xyxy"] = [x - 5, y - 5, x + 5, y + 5]
            else:
                record["bbox_xyxy"] = []
            mask_records.append(record)

        masks_path.write_text(json.dumps(mask_records, indent=2))
        caption_path.write_text("; ".join(captions) if captions else "Construction site scene.")
        return True

    except Exception as e:
        print(f"  ERROR annotating {image_path.name}: {e}")
        return False


def run_pipeline(
    frames_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    captioner: str = "florence",
) -> None:
    fallback = False
    depth_estimator = None
    localizer = None

    try:
        print("Loading DepthEstimator (ONNX via VQASynth)…")
        from vqasynth.depth import DepthEstimator
        depth_estimator = DepthEstimator(from_onnx=True)
        print(f"Loading Localizer (captioner={captioner}, segmenter=SAM2)…")
        from vqasynth.localize import Localizer
        localizer = Localizer(captioner_type=captioner)
    except ImportError:
        print("VQASynth not available — using transformers fallback (relative depth, DETR).")
        depth_estimator = _FallbackAnnotator()
        fallback = True

    frame_paths = sorted(frames_dir.rglob("*.jpg")) + sorted(frames_dir.rglob("*.png"))
    if limit:
        frame_paths = frame_paths[:limit]

    ok = errors = 0
    for img_path in frame_paths:
        ann_out = out_dir / img_path.stem
        print(f"  annotating {img_path.name}…", end=" ", flush=True)
        success = annotate_frame(img_path, ann_out, depth_estimator, localizer, fallback)
        if success:
            ok += 1
            print("ok")
        else:
            errors += 1

    print(f"\nDone: {ok} annotated, {errors} errors")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate extracted frames with depth + segmentation.")
    parser.add_argument("--frames_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--captioner", choices=["florence", "molmo"], default="florence")
    args = parser.parse_args()

    run_pipeline(args.frames_dir, args.out_dir, args.limit, args.captioner)


if __name__ == "__main__":
    main()
