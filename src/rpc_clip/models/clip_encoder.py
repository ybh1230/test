from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None


def _resize_positional_embedding(pos_embed: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    cls_pos = pos_embed[:1]
    patch_pos = pos_embed[1:]
    old_grid = int(patch_pos.shape[0] ** 0.5)
    if old_grid * old_grid != patch_pos.shape[0]:
        raise ValueError("Cannot infer square CLIP positional grid.")
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=grid_hw, mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, patch_pos.shape[1])
    return torch.cat([cls_pos, patch_pos], dim=0)


class FrozenOpenCLIPViT(nn.Module):
    def __init__(self, model_name: str, pretrained: str, prompt_templates: Iterable[str]) -> None:
        super().__init__()
        self.is_toy = model_name.lower() == "toy" or pretrained.lower() == "toy"
        self.prompt_templates = list(prompt_templates)
        if self.is_toy:
            self.embed_dim = 64
            self.toy_conv = nn.Conv2d(3, self.embed_dim, kernel_size=16, stride=16, bias=False)
            torch.manual_seed(123)
            nn.init.normal_(self.toy_conv.weight, std=0.02)
            for param in self.toy_conv.parameters():
                param.requires_grad_(False)
            return

        if open_clip is None:
            raise ImportError("Install open_clip_torch for real RPC-CLIP training, or use --smoke for a toy check.")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        visual = self.clip_model.visual
        if not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
            raise ValueError("RPC-CLIP currently expects an OpenCLIP ViT visual backbone.")
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is None:
            output_dim = visual.proj.shape[-1] if getattr(visual, "proj", None) is not None else visual.ln_post.normalized_shape[0]
        self.embed_dim = int(output_dim)

    @torch.no_grad()
    def encode_text_classes(self, class_names: List[str], device: torch.device) -> torch.Tensor:
        if self.is_toy:
            generator = torch.Generator(device="cpu").manual_seed(456)
            text = torch.randn(len(class_names), self.embed_dim, generator=generator)
            return F.normalize(text, dim=-1).to(device)
        features = []
        model = self.clip_model.to(device)
        for class_name in class_names:
            clean_name = class_name.replace("_", " ")
            texts = [template.format(clean_name) for template in self.prompt_templates]
            tokens = self.tokenizer(texts).to(device)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
            features.append(F.normalize(text_features.mean(dim=0), dim=0))
        return torch.stack(features, dim=0)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.is_toy:
            x = self.toy_conv(images.float())
            grid_hw = (x.shape[-2], x.shape[-1])
            tokens = x.flatten(2).transpose(1, 2)
            return F.normalize(tokens, dim=-1), grid_hw
        visual = self.clip_model.visual
        dtype = next(visual.parameters()).dtype
        x = images.to(dtype=dtype)
        x = visual.conv1(x)
        grid_hw = (x.shape[-2], x.shape[-1])
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)

        pos_embed = visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if pos_embed.shape[0] != x.shape[1]:
            pos_embed = _resize_positional_embedding(pos_embed, grid_hw)
        x = x + pos_embed

        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        tokens = x[:, 1:, :]
        if visual.proj is not None:
            tokens = tokens @ visual.proj
        tokens = F.normalize(tokens.float(), dim=-1)
        return tokens, grid_hw
