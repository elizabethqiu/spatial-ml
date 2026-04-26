"""
Extract frames from construction videos at a target FPS,
filter with CLIP embeddings using VQASynth's TagFilter approach
(falls back to transformers CLIP when VQASynth is not installed).
"""

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

INCLUDE_TAGS = [
    "construction site",
    "heavy equipment",
    "excavator",
    "crane",
    "bulldozer",
    "workers on a job site",
    "outdoor construction",
]
EXCLUDE_TAGS = [
    "indoor",
    "office",
    "blank",
    "text document",
]


# ---------------------------------------------------------------------------
# CLIP filter — VQASynth if available, otherwise transformers fallback
# ---------------------------------------------------------------------------

class _CLIPFilter:
    def __init__(self):
        try:
            from vqasynth.embeddings import EmbeddingGenerator, TagFilter
            self._embedder = EmbeddingGenerator()
            self._tagger = TagFilter()
            self._mode = "vqasynth"
            print("CLIP filter: using VQASynth")
        except ImportError:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._torch = torch
            self._mode = "transformers"
            print("CLIP filter: using transformers (VQASynth not available)")

    def keep(self, image: Image.Image) -> bool:
        if self._mode == "vqasynth":
            emb = self._embedder.run(image)
            best = self._tagger.get_best_matching_tag(emb, INCLUDE_TAGS + EXCLUDE_TAGS)
            return self._tagger.filter_by_tag(best, INCLUDE_TAGS, EXCLUDE_TAGS)

        # transformers fallback: zero-shot classification
        all_tags = INCLUDE_TAGS + EXCLUDE_TAGS
        inputs = self._processor(
            text=all_tags, images=image, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]
        probs = logits.softmax(dim=0).tolist()
        n_include = len(INCLUDE_TAGS)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return best_idx < n_include  # include tags come first


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 2.0,
    clip_filter: bool = True,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(src_fps / target_fps))

    clf = _CLIPFilter() if clip_filter else None

    metadata = []
    frame_idx = 0
    kept = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keep = clf.keep(pil) if clf is not None else True

            if keep:
                out_path = out_dir / f"{video_path.stem}_f{frame_idx:06d}.jpg"
                pil.save(out_path, quality=90)
                metadata.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": round(frame_idx / src_fps, 3),
                    "path": str(out_path),
                })
                kept += 1

        frame_idx += 1

    cap.release()
    print(f"{video_path.name}: {kept}/{frame_idx} frames kept")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--no_clip", action="store_true")
    args = parser.parse_args()

    all_meta = {}
    for video in sorted(args.video_dir.glob("*.mp4")):
        meta = extract_frames(video, args.out_dir / video.stem, args.fps, not args.no_clip)
        all_meta[video.name] = meta

    meta_path = args.out_dir / "metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2))
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
