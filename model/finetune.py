"""
Finetune SpatialVLM on the SpatialVLM data mixture + construction VQA pairs,
following the two-stage PaLM-E-2 recipe:

  Stage 1 — freeze image encoder, train on SpatialVLM mixture + construction VQA
  Stage 2 — unfreeze image encoder, short run on construction data only
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpatialVQADataset(Dataset):
    """
    Expects a directory of JSON files (one per frame) each containing a list of
    {"question": ..., "answer": ..., "image_path": ...} dicts.
    """

    def __init__(self, vqa_dir: Path, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.samples: list[dict] = []
        for jf in sorted(vqa_dir.glob("*.json")):
            pairs = json.loads(jf.read_text())
            for pair in pairs:
                if "image_path" in pair:
                    self.samples.append(pair)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        prompt = f"Question: {s['question']}\nAnswer:"
        target = s["answer"]

        enc = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).input_ids

        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": labels.squeeze(0)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def freeze_image_encoder(model):
    for name, param in model.named_parameters():
        if "vision" in name or "image_encoder" in name:
            param.requires_grad = False


def unfreeze_image_encoder(model):
    for param in model.parameters():
        param.requires_grad = True


def train_stage(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epochs: int,
    stage_name: str,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        print(f"[{stage_name}] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="remyxai/SpaceLLaVA")
    parser.add_argument("--vqa_dir", type=Path, default=Path("data/vqa_pairs"))
    parser.add_argument("--construction_vqa_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/spatialvlm-construction"))
    parser.add_argument("--stage1_epochs", type=int, default=3)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Stage 1
    freeze_image_encoder(model)
    ds1 = SpatialVQADataset(args.vqa_dir, processor)
    dl1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=4)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sched1 = get_cosine_schedule_with_warmup(opt1, 100, args.stage1_epochs * len(dl1))
    train_stage(model, dl1, opt1, sched1, args.device, args.stage1_epochs, "stage1")

    # Stage 2 — construction data only, unfreeze encoder
    if args.construction_vqa_dir and args.construction_vqa_dir.exists():
        unfreeze_image_encoder(model)
        ds2 = SpatialVQADataset(args.construction_vqa_dir, processor)
        dl2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=4)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr / 5)
        sched2 = get_cosine_schedule_with_warmup(opt2, 20, args.stage2_epochs * len(dl2))
        train_stage(model, dl2, opt2, sched2, args.device, args.stage2_epochs, "stage2")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
