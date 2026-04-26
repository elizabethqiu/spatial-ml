"""
Upload the spatial-ml dataset to Hugging Face.

Uploads:
  data/vqa_pairs/   → parquet dataset (questions + answers + image paths)
  data/frames/      → images (as dataset image column)
  data/annotations/ → depth + masks (as dataset files)

Usage:
  python pipeline/upload_dataset.py --repo elizqiu/spatial-ml-construction
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_dataset(repo_id: str, token: str, private: bool = True):
    api = HfApi(token=token)

    print(f"Creating dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)

    # --- 1. VQA pairs as a single JSONL file ---
    vqa_dir = Path("data/vqa_pairs")
    if vqa_dir.exists():
        print("Combining VQA pairs into dataset.jsonl…")
        all_pairs = []
        for jf in sorted(vqa_dir.glob("*.json")):
            try:
                pairs = json.loads(jf.read_text())
                all_pairs.extend(pairs)
            except Exception:
                pass
        jsonl_path = Path("/tmp/spatial_ml_vqa.jsonl")
        with jsonl_path.open("w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"  {len(all_pairs)} QA pairs → uploading…")
        api.upload_file(
            path_or_fileobj=str(jsonl_path),
            path_in_repo="data/vqa_pairs.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add VQA pairs",
        )
        print("  ✓ vqa_pairs.jsonl uploaded")

    # --- 2. Frames (images) ---
    frames_dir = Path("data/frames")
    if frames_dir.exists():
        frame_files = sorted(frames_dir.rglob("*.jpg"))
        print(f"Uploading {len(frame_files)} frames…")
        api.upload_folder(
            folder_path=str(frames_dir),
            path_in_repo="data/frames",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add extracted frames",
            ignore_patterns=["*.DS_Store"],
        )
        print("  ✓ frames uploaded")

    # --- 3. Annotations (depth + masks + captions) ---
    ann_dir = Path("data/annotations")
    if ann_dir.exists():
        print("Uploading annotations (depth, masks, captions)…")
        api.upload_folder(
            folder_path=str(ann_dir),
            path_in_repo="data/annotations",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add frame annotations",
            ignore_patterns=["*.DS_Store"],
        )
        print("  ✓ annotations uploaded")

    print(f"\nDataset live at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="elizqiu/spatial-ml-construction")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — run: export $(grep -v '^#' .env | xargs)")

    upload_dataset(args.repo, token, private=not args.public)


if __name__ == "__main__":
    main()
