from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from rpc_clip.data.voc import VOC_CLASSES
from rpc_clip.models.clip_encoder import FrozenOpenCLIPViT
from rpc_clip.models.decoder import PatchDecoder


class RPCClip(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.num_fg_classes = len(VOC_CLASSES)
        self.num_classes = self.num_fg_classes + 1
        self.text_temperature = float(model_cfg["text_temperature"])
        self.prototype_temperature = float(model_cfg["prototype_temperature"])
        self.prototype_momentum = float(config["rpc"]["prototype_momentum"])
        self.update_confidence = float(config["rpc"]["update_confidence"])
        self.text_weight = float(config["rpc"]["text_weight"])
        self.prototype_weight = float(config["rpc"]["prototype_weight"])
        self.decoder_weight = float(config["rpc"]["decoder_weight"])
        self.reliability_mode = str(config["rpc"].get("reliability_mode", "entropy"))
        weight_sum = self.text_weight + self.prototype_weight + self.decoder_weight
        self.text_weight /= weight_sum
        self.prototype_weight /= weight_sum
        self.decoder_weight /= weight_sum

        self.clip = FrozenOpenCLIPViT(
            model_cfg["clip_model"],
            model_cfg["clip_pretrained"],
            model_cfg["prompt_templates"],
        )
        self.decoder = PatchDecoder(
            embed_dim=self.clip.embed_dim,
            num_classes=self.num_classes,
            hidden_dim=int(model_cfg["decoder_hidden"]),
            dropout=float(model_cfg["decoder_dropout"]),
        )
        self.register_buffer("text_features", torch.empty(self.num_fg_classes, self.clip.embed_dim), persistent=True)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.clip.embed_dim), persistent=True)
        self.register_buffer("prototype_seen", torch.zeros(self.num_classes), persistent=True)

    def initialize_text_features(self, device: torch.device) -> None:
        text_features = self.clip.encode_text_classes(VOC_CLASSES, device)
        self.text_features.copy_(text_features.to(self.text_features.device))
        self.prototypes[1:].copy_(text_features.to(self.prototypes.device))
        self.prototype_seen[1:].fill_(1.0)

    def rpc_state_dict(self) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        return {
            "decoder": self.decoder.state_dict(),
            "text_features": self.text_features.detach().cpu(),
            "prototypes": self.prototypes.detach().cpu(),
            "prototype_seen": self.prototype_seen.detach().cpu(),
        }

    def load_rpc_state_dict(self, state: Dict) -> None:
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"], strict=True)
            self.text_features.copy_(state["text_features"].to(self.text_features.device))
            self.prototypes.copy_(state["prototypes"].to(self.prototypes.device))
            self.prototype_seen.copy_(state["prototype_seen"].to(self.prototype_seen.device))
        else:
            self.load_state_dict(state, strict=True)

    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, tuple[int, int]]:
        return self.clip(images)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor | tuple[int, int]]:
        tokens, grid_hw = self.encode_images(images)
        logits = self.decoder(tokens, grid_hw)
        return {"tokens": tokens, "grid_hw": grid_hw, "logits": logits}

    def _mask_absent_foreground(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        absent = labels <= 0
        masked = masked.masked_fill(absent[:, None, :], -1e4)
        return masked

    def text_logits(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        fg_logits = (tokens @ self.text_features.t()) / self.text_temperature
        fg_logits = self._mask_absent_foreground(fg_logits, labels)
        max_fg = fg_logits.max(dim=-1, keepdim=True).values
        bg_logits = -max_fg.clamp(min=-12.0, max=12.0)
        return torch.cat([bg_logits, fg_logits], dim=-1)

    def prototype_logits(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = (tokens @ prototypes.t()) / self.prototype_temperature
        fg_logits = self._mask_absent_foreground(logits[:, :, 1:], labels)
        return torch.cat([logits[:, :, :1], fg_logits], dim=-1)

    def pseudo_targets(
        self,
        tokens: torch.Tensor,
        decoder_logits: torch.Tensor,
        labels: torch.Tensor,
        confidence_threshold: float,
    ) -> Dict[str, torch.Tensor]:
        grid_h, grid_w = decoder_logits.shape[-2:]
        text_logits = self.text_logits(tokens, labels)
        proto_logits = self.prototype_logits(tokens, labels)
        dec_logits = decoder_logits.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1, self.num_classes)

        text_prob = text_logits.softmax(dim=-1)
        proto_prob = proto_logits.softmax(dim=-1)
        dec_prob = dec_logits.detach().softmax(dim=-1)
        joint = self.text_weight * text_prob + self.prototype_weight * proto_prob + self.decoder_weight * dec_prob
        confidence, target = joint.max(dim=-1)

        if self.reliability_mode == "uniform":
            reliability = torch.ones_like(confidence)
        elif self.reliability_mode == "confidence":
            reliability = confidence
        elif self.reliability_mode == "entropy":
            entropy = -(joint.clamp_min(1e-7) * joint.clamp_min(1e-7).log()).sum(dim=-1)
            reliability = 1.0 - entropy / torch.log(torch.tensor(float(self.num_classes), device=joint.device))
            reliability = reliability.clamp(0.0, 1.0) * confidence
        else:
            raise ValueError(f"Unknown reliability_mode: {self.reliability_mode}")

        valid = confidence >= confidence_threshold
        target = target.masked_fill(~valid, 255)
        return {
            "target": target.reshape(-1, grid_h, grid_w),
            "reliability": reliability.reshape(-1, grid_h, grid_w),
            "confidence": confidence.reshape(-1, grid_h, grid_w),
            "joint_prob": joint.reshape(-1, grid_h, grid_w, self.num_classes).permute(0, 3, 1, 2),
        }

    @torch.no_grad()
    def update_prototypes(self, tokens: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> None:
        flat_tokens = tokens.reshape(-1, tokens.shape[-1])
        flat_target = target.reshape(-1)
        flat_conf = confidence.reshape(-1)
        for class_id in range(self.num_classes):
            mask = (flat_target == class_id) & (flat_conf >= self.update_confidence)
            if mask.sum() < 4:
                continue
            proto = F.normalize(flat_tokens[mask].mean(dim=0), dim=0)
            if self.prototype_seen[class_id] < 0.5:
                self.prototypes[class_id].copy_(proto)
                self.prototype_seen[class_id] = 1.0
            else:
                updated = self.prototype_momentum * self.prototypes[class_id] + (1.0 - self.prototype_momentum) * proto
                self.prototypes[class_id].copy_(F.normalize(updated, dim=0))


def boundary_aware_smoothness(logits: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    probs = F.interpolate(probs, size=images.shape[-2:], mode="bilinear", align_corners=False)
    dx_prob = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()
    dy_prob = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
    dx_img = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dy_img = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    weight_x = torch.exp(-4.0 * dx_img)
    weight_y = torch.exp(-4.0 * dy_img)
    return (dx_prob * weight_x).mean() + (dy_prob * weight_y).mean()


def image_level_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    fg_logits = logits[:, 1:].amax(dim=(-2, -1))
    avg_logits = logits[:, 1:].mean(dim=(-2, -1))
    pooled = 0.7 * fg_logits + 0.3 * avg_logits
    return F.binary_cross_entropy_with_logits(pooled, labels)


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
    valid = (target != 255).float()
    weight = reliability.detach() * valid
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def consistency_loss(logits: torch.Tensor, joint_prob: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    log_prob = logits.log_softmax(dim=1)
    loss = F.kl_div(log_prob, joint_prob.detach(), reduction="none").sum(dim=1)
    weight = reliability.detach()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
