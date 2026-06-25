from __future__ import annotations

import argparse
import copy
import csv
import shutil
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from rpc_clip.data.voc import VOCDataset, VOC_CLASSES, worker_init_fn
from rpc_clip.engine.evaluate import evaluate
from rpc_clip.models.rpc import (
    RPCClip,
    boundary_aware_smoothness,
    consistency_loss,
    image_level_loss,
    weighted_ce,
)
from rpc_clip.utils.config import apply_overrides, load_config, save_config, save_jsonl, set_seed, to_device


def _build_loader(config: dict, split: str, train: bool, smoke: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = VOCDataset(
        root=data_cfg["root"],
        split=split,
        image_size=int(data_cfg["image_size"]),
        train=train,
        cls_label_file=data_cfg.get("cls_label_file"),
        smoke=smoke,
        smoke_size=16 if train else 8,
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=train,
        num_workers=0 if smoke else int(data_cfg["workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=worker_init_fn if train else None,
    )


def _write_csv_header(path: Path) -> None:
    if path.is_file():
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "loss_ce", "loss_cls", "loss_cons", "loss_boundary", "val_miou", "time_sec"])


def train(config: dict, output: Path, smoke: bool = False) -> None:
    if smoke:
        config = copy.deepcopy(config)
        config["model"]["clip_model"] = "toy"
        config["model"]["clip_pretrained"] = "toy"
        config["model"]["decoder_hidden"] = 32
        config["data"]["batch_size"] = 4
        config["train"]["log_interval"] = 1
    set_seed(int(config.get("seed", 7)))
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _build_loader(config, config["data"]["train_split"], train=True, smoke=smoke)
    val_loader = _build_loader(config, config["data"]["val_split"], train=False, smoke=smoke)

    model = RPCClip(config).to(device)
    model.initialize_text_features(device)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")
    csv_path = output / "metrics.csv"
    _write_csv_header(csv_path)

    best_miou = -1.0
    epochs = 1 if smoke else int(config["train"]["epochs"])
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "cls": 0.0, "cons": 0.0, "boundary": 0.0}
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = model(batch["image"])
                pseudo = model.pseudo_targets(
                    out["tokens"],
                    out["logits"],
                    batch["label"],
                    confidence_threshold=float(config["rpc"]["pseudo_confidence"]),
                )
                loss_ce = weighted_ce(out["logits"], pseudo["target"], pseudo["reliability"])
                loss_cls = image_level_loss(out["logits"], batch["label"])
                loss_cons = consistency_loss(out["logits"], pseudo["joint_prob"], pseudo["reliability"])
                loss_boundary = boundary_aware_smoothness(out["logits"], batch["image"])
                loss = (
                    float(config["train"]["lambda_ce"]) * loss_ce
                    + float(config["train"]["lambda_cls"]) * loss_cls
                    + float(config["train"]["lambda_consistency"]) * loss_cons
                    + float(config["train"]["lambda_boundary"]) * loss_boundary
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_prototypes(out["tokens"].detach(), pseudo["target"].detach(), pseudo["confidence"].detach())

            running["loss"] += float(loss.detach())
            running["ce"] += float(loss_ce.detach())
            running["cls"] += float(loss_cls.detach())
            running["cons"] += float(loss_cons.detach())
            running["boundary"] += float(loss_boundary.detach())
            if step % int(config["train"]["log_interval"]) == 0 or step == 1:
                progress.set_postfix(loss=running["loss"] / step)

        denom = max(len(train_loader), 1)
        eval_scores = {"miou": -1.0}
        if epoch % int(config["train"]["eval_interval"]) == 0:
            eval_scores = evaluate(model, val_loader, device, int(config["eval"]["ignore_index"]))
            if eval_scores["miou"] > best_miou:
                best_miou = float(eval_scores["miou"])
                torch.save(
                    {
                        "rpc_state": model.rpc_state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "best_miou": best_miou,
                        "classes": VOC_CLASSES,
                    },
                    output / "best.pt",
                )

        row = [
            epoch,
            running["loss"] / denom,
            running["ce"] / denom,
            running["cls"] / denom,
            running["cons"] / denom,
            running["boundary"] / denom,
            eval_scores["miou"],
            time.time() - start_time,
        ]
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        save_jsonl(
            {
                "epoch": epoch,
                "train_loss": row[1],
                "loss_ce": row[2],
                "loss_cls": row[3],
                "loss_consistency": row[4],
                "loss_boundary": row[5],
                "val_miou": eval_scores["miou"],
                "best_miou": best_miou,
            },
            output / "metrics.jsonl",
        )

    torch.save({"rpc_state": model.rpc_state_dict(), "config": config, "epoch": epochs, "best_miou": best_miou}, output / "last.pt")
    if (output / "best.pt").is_file():
        shutil.copy2(output / "best.pt", output / "checkpoint_for_visualization.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RPC-CLIP on a single GPU.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/rpc_clip_voc")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic sanity check.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, help="Override config keys, e.g. train.epochs=4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    train(config, Path(args.output), smoke=args.smoke)
