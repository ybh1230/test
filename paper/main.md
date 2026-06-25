# Reliability-Guided Prompt-Prototype Calibration for Weakly Supervised Semantic Segmentation

## Abstract

Weakly supervised semantic segmentation (WSSS) seeks to learn dense masks from
image-level labels, but modern CLIP-based pipelines still suffer from noisy
patch-text activations, incomplete object coverage, and unstable background
assignment. We propose **RPC-CLIP**, a single-GPU friendly framework that freezes
CLIP and trains only a lightweight patch decoder. The key idea is to calibrate
CLIP pseudo labels with a momentum prototype bank and an entropy-based
reliability estimator. Given image-level labels, RPC-CLIP first derives
class-constrained patch scores from prompt-averaged CLIP text embeddings. It then
updates foreground and background prototypes from high-confidence patches, fuses
text, prototype, and decoder distributions according to reliability, and applies
a boundary-aware smoothness regularizer.

**TBD:** Fill final mIoU, test server numbers, runtime, and significance after
real experiments.

## 1. Introduction

Weakly supervised semantic segmentation reduces annotation cost by replacing
pixel-level masks with cheaper supervision such as image-level labels. Classic
CAM-based methods localize discriminative object parts, but they often miss
complete object extents. Recent CLIP-based methods offer stronger open-vocabulary
semantic priors, yet patch-text similarity maps remain noisy: confident patches
may cover only the most recognizable parts, while background pixels can be
mistakenly attracted by foreground prompts.

This paper asks whether a frozen CLIP model can be made more reliable without
introducing a heavy segmentation backbone. Our answer is RPC-CLIP, a
prompt-prototype calibration framework designed for fast single-GPU training.
Instead of fine-tuning CLIP, we keep the visual-language encoder fixed and train
a small patch decoder on pseudo labels. The pseudo labels are not taken directly
from CLIP. They are calibrated by a momentum prototype bank that accumulates
class evidence from confident patches and by a reliability score that suppresses
uncertain regions.

Contributions:

- A lightweight frozen-CLIP WSSS framework that trains on a single GPU and uses
  only image-level labels.
- Prompt-prototype calibration, where class-constrained CLIP text scores are
  corrected by a momentum prototype bank.
- Reliability-guided pseudo-label fusion combining text, prototype, and decoder
  distributions.
- A complete experimental protocol with comparison, ablation, and visualization
  templates for PASCAL VOC 2012.

## 2. Related Work

**Weakly supervised semantic segmentation.** CAM-based WSSS methods use image
classifiers to localize discriminative regions. SEAM improves consistency under
image transformations, while reliability-aware methods show that mining
trustworthy regions is crucial for end-to-end WSSS. RPC-CLIP follows this
reliability-oriented line but replaces a supervised classification backbone with
a frozen CLIP encoder.

**Vision-language models for segmentation.** CLIP transfers language supervision
to visual recognition and has become a strong prior for weakly supervised
segmentation. CLIP-ES and WeCLIP demonstrate that CLIP patch-text similarities
can produce useful pseudo masks. However, raw language similarity is not
calibrated for dense prediction. Our method uses text features as initialization
and continuously refines them with image-level visual prototypes.

**Prototype learning.** Prototype-based segmentation summarizes class appearance
with representative features. In weak supervision, prototype updates must be
conservative because incorrect pseudo labels can quickly contaminate the class
representation. RPC-CLIP addresses this issue with confidence-gated momentum
updates and entropy-based reliability weighting.

## 3. Method

Given an image and image-level labels, a frozen CLIP ViT extracts patch features.
A lightweight decoder predicts background plus foreground logits. Only the
decoder is optimized.

### Prompt-Constrained CLIP Seeds

Class text embeddings are averaged across prompt templates. Patch-text logits are
computed with cosine similarity, and absent classes are masked according to
image-level labels.

### Momentum Prototype Calibration

RPC-CLIP maintains background and foreground prototypes. Foreground prototypes
are initialized from text embeddings. Background and foreground prototypes are
updated from high-confidence patches using a momentum rule.

### Reliability-Guided Fusion

The pseudo distribution is the weighted fusion of text probability, prototype
probability, and decoder probability. Reliability is confidence multiplied by
one minus normalized entropy. Low-confidence pixels are ignored.

The final loss combines reliability-weighted cross entropy, image-level
classification loss, KL consistency, and boundary-aware smoothness.

## 4. Experiments

### Datasets and Metrics

Use PASCAL VOC 2012 with image-level labels for training. Report mIoU on val and
test. Optional MS COCO can be added after VOC is stable.

### Implementation Details

Default fast setting:

- CLIP ViT-B/16
- input 224 x 224
- batch size 16
- AdamW
- 8 epochs
- frozen CLIP, train decoder only
- single GPU

### Comparison Table

| Method | Backbone | Val mIoU | Test mIoU |
|---|---:|---:|---:|
| CAM | VGG16 | TBD | TBD |
| SEAM | ResNet38 | TBD | TBD |
| RRM | ResNet38 | TBD | TBD |
| CLIP-ES | CLIP ViT-B/16 | TBD | TBD |
| SFC | CLIP ViT-B/16 | TBD | TBD |
| WeCLIP | CLIP ViT-B/16 | TBD | TBD |
| RPC-CLIP | CLIP ViT-B/16 | TBD | TBD |

### Ablation Table

| Variant | Val mIoU | Delta |
|---|---:|---:|
| Text-only pseudo labels | TBD | -- |
| + Prototype calibration | TBD | TBD |
| + Reliability weighting | TBD | TBD |
| + Boundary smoothness (RPC-CLIP) | TBD | TBD |

### Qualitative Results

Use `visualize.py` to generate panels containing input image, ground truth,
prediction, and overlay. Include both success and failure cases.

## 5. Conclusion

RPC-CLIP is a fast frozen-CLIP framework for weakly supervised semantic
segmentation. By combining prompt-constrained language evidence, momentum
prototype calibration, and reliability-guided pseudo-label fusion, it aims to
improve pseudo-mask quality while keeping training practical on a single GPU.

