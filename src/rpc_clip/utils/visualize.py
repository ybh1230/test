from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
if os.name == "nt" and "WINDIR" not in os.environ:
    os.environ["WINDIR"] = r"C:\Windows"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rpc_clip.data.voc import CLIP_MEAN, CLIP_STD, VOC_CLASSES, VOC_COLORMAP, VOCDataset
from rpc_clip.models.rpc import RPCClip
from rpc_clip.utils.config import apply_overrides, load_config, to_device


def _denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CLIP_MEAN, device=image.device)[:, None, None]
    std = torch.tensor(CLIP_STD, device=image.device)[:, None, None]
    image = (image * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).detach().cpu().numpy()


def _colorize(mask: np.ndarray) -> np.ndarray:
    mask = mask.copy()
    mask[mask == 255] = 0
    mask = np.clip(mask, 0, len(VOC_COLORMAP) - 1)
    return VOC_COLORMAP[mask]


def save_panel(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, out_path: Path, title: str) -> None:
    image_np = _denormalize(image)
    gt_np = gt.detach().cpu().numpy().astype(np.uint8)
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(_colorize(gt_np))
    axes[1].set_title("GT")
    axes[2].imshow(_colorize(pred_np))
    axes[2].set_title("Prediction")
    axes[3].imshow(image_np)
    axes[3].imshow(_colorize(pred_np), alpha=0.48)
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def visualize(config: dict, checkpoint_path: str | Path, output: Path, split: str, limit: int, smoke: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VOCDataset(
        root=config["data"]["root"],
        split=split,
        image_size=int(config["data"]["image_size"]),
        train=False,
        cls_label_file=config["data"].get("cls_label_file"),
        smoke=smoke,
        smoke_size=max(limit, 8),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = RPCClip(config).to(device)
    model.initialize_text_features(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_rpc_state_dict(checkpoint.get("rpc_state", checkpoint.get("model")))
    model.eval()

    names = ["background"] + VOC_CLASSES
    output.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        batch = to_device(batch, device)
        out = model(batch["image"])
        eval_cfg = config.get("eval", {})
        patch_logits = model.inference_logits(
            out["tokens"],
            out["logits"],
            mode=str(eval_cfg.get("inference_mode", "decoder")),
            class_prior_topk=int(eval_cfg.get("class_prior_topk", 3)),
            text_weight=float(eval_cfg.get("inference_text_weight", 0.45)),
            prototype_weight=float(eval_cfg.get("inference_prototype_weight", 0.20)),
            decoder_weight=float(eval_cfg.get("inference_decoder_weight", 0.35)),
        )
        logits = F.interpolate(patch_logits, size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)[0]
        present = torch.unique(pred).detach().cpu().tolist()
        present_names = ", ".join(names[class_id] for class_id in present if class_id < len(names))
        save_panel(
            batch["image"][0],
            batch["mask"][0],
            pred,
            output / f"{batch['id'][0]}_panel.png",
            f"{batch['id'][0]} | {present_names}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save RPC-CLIP prediction panels.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default="runs/rpc_clip_voc/visuals")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    visualize(config, args.checkpoint, Path(args.output), args.split, args.limit, smoke=args.smoke)
