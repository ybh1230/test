from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentationMeter:
    num_classes: int
    ignore_index: int = 255

    def __post_init__(self) -> None:
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

    def reset(self) -> None:
        self.hist.fill(0)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        target_np = target.detach().cpu().numpy().astype(np.int64)
        mask = (target_np != self.ignore_index) & (target_np >= 0) & (target_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * target_np[mask] + pred_np[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.hist += hist

    def scores(self) -> dict[str, float | list[float]]:
        inter = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - inter
        iou = inter / np.maximum(union, 1.0)
        acc = inter.sum() / np.maximum(self.hist.sum(), 1.0)
        return {
            "miou": float(np.nanmean(iou)),
            "pixel_acc": float(acc),
            "iou": [float(x) for x in iou],
        }

