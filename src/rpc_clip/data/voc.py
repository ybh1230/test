from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]

VOC_COLORMAP = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _read_split(root: Path, split: str) -> List[str]:
    split_path = Path(split)
    candidates = [
        split_path if split_path.is_file() else None,
        root / "ImageSets" / "Segmentation" / f"{split}.txt",
        root / "ImageSets" / "SegmentationAug" / f"{split}.txt",
        root / "ImageSets" / "Main" / f"{split}.txt",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as handle:
                return [line.strip().split()[0] for line in handle if line.strip()]
    raise FileNotFoundError(f"Could not find split file for '{split}' under {root}")


def _load_cls_labels(path: Optional[str | Path]) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=True).item()
    elif path.suffix == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raise ValueError(f"Unsupported class label file: {path}")
    labels: Dict[str, np.ndarray] = {}
    for key, value in raw.items():
        labels[str(key)] = np.asarray(value, dtype=np.float32)
    return labels


def _labels_from_xml(xml_path: Path) -> np.ndarray:
    label = np.zeros(len(VOC_CLASSES), dtype=np.float32)
    if not xml_path.is_file():
        return label
    tree = ET.parse(xml_path)
    for obj in tree.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name in VOC_CLASSES:
            label[VOC_CLASSES.index(name)] = 1.0
    return label


def _labels_from_mask(mask_path: Path) -> np.ndarray:
    label = np.zeros(len(VOC_CLASSES), dtype=np.float32)
    if not mask_path.is_file():
        return label
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    for class_id in np.unique(mask):
        if 1 <= int(class_id) <= len(VOC_CLASSES):
            label[int(class_id) - 1] = 1.0
    return label


def _mask_path(root: Path, image_id: str) -> Optional[Path]:
    for folder in ("SegmentationClass", "SegmentationClassAug"):
        candidate = root / folder / f"{image_id}.png"
        if candidate.is_file():
            return candidate
    return None


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        train: bool,
        cls_label_file: Optional[str | Path] = None,
        smoke: bool = False,
        smoke_size: int = 32,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train = train
        self.smoke = smoke
        self.cls_labels = _load_cls_labels(cls_label_file)
        self.ids = [f"smoke_{idx:04d}" for idx in range(smoke_size)] if smoke else _read_split(self.root, split)

        if train:
            self.image_tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )
        else:
            self.image_tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(CLIP_MEAN, CLIP_STD),
                ]
            )
        self.mask_tf = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.ids)

    def _smoke_sample(self, index: int) -> Dict[str, torch.Tensor | str]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        image = transforms.Normalize(CLIP_MEAN, CLIP_STD)(image)
        label = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        label[index % len(VOC_CLASSES)] = 1.0
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
        y0 = self.image_size // 4
        y1 = y0 + self.image_size // 2
        mask[y0:y1, y0:y1] = (index % len(VOC_CLASSES)) + 1
        return {"image": image, "label": label, "mask": mask, "id": self.ids[index]}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        if self.smoke:
            return self._smoke_sample(index)

        image_id = self.ids[index]
        image_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_tf(image)

        if self.cls_labels and image_id in self.cls_labels:
            label = self.cls_labels[image_id]
        else:
            label = _labels_from_xml(self.root / "Annotations" / f"{image_id}.xml")
            if label.sum() == 0:
                mask_candidate = _mask_path(self.root, image_id)
                if mask_candidate is not None:
                    label = _labels_from_mask(mask_candidate)

        mask_tensor = torch.full((self.image_size, self.image_size), 255, dtype=torch.long)
        mask_candidate = _mask_path(self.root, image_id)
        if mask_candidate is not None:
            mask = Image.open(mask_candidate)
            mask = self.mask_tf(mask)
            mask_tensor = torch.as_tensor(np.asarray(mask, dtype=np.uint8), dtype=torch.long)

        return {
            "image": image_tensor,
            "label": torch.as_tensor(label, dtype=torch.float32),
            "mask": mask_tensor,
            "id": image_id,
        }


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
