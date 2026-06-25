from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rpc_clip.data.voc import VOCDataset, VOC_CLASSES
from rpc_clip.models.rpc import RPCClip
from rpc_clip.utils.config import apply_overrides, load_config, save_jsonl, to_device
from rpc_clip.utils.metrics import SegmentationMeter


@torch.no_grad()
def evaluate(model: RPCClip, loader: DataLoader, device: torch.device, ignore_index: int = 255) -> dict:
    model.eval()
    meter = SegmentationMeter(num_classes=len(VOC_CLASSES) + 1, ignore_index=ignore_index)
    for batch in tqdm(loader, desc="eval", dynamic_ncols=True, leave=False):
        batch = to_device(batch, device)
        out = model(batch["image"])
        logits = F.interpolate(out["logits"], size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)
        meter.update(pred, batch["mask"])
    return meter.scores()


def _build_eval_loader(config: dict, split: Optional[str], smoke: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = VOCDataset(
        root=data_cfg["root"],
        split=split or data_cfg["val_split"],
        image_size=int(data_cfg["image_size"]),
        train=False,
        cls_label_file=data_cfg.get("cls_label_file"),
        smoke=smoke,
        smoke_size=8,
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=0 if smoke else int(data_cfg["workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an RPC-CLIP checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RPCClip(config).to(device)
    model.initialize_text_features(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_rpc_state_dict(checkpoint.get("rpc_state", checkpoint.get("model")))
    loader = _build_eval_loader(config, args.split, args.smoke)
    scores = evaluate(model, loader, device, int(config["eval"]["ignore_index"]))
    print(scores)
    if args.output:
        save_jsonl(scores, Path(args.output) / "eval.jsonl")
