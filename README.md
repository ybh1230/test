# AAAI 2027 RPC-CLIP Starter

This workspace contains a paper-and-code scaffold for a fast single-GPU weakly
supervised semantic segmentation project:

**RPC-CLIP: Reliability-Guided Prompt-Prototype Calibration for
Weakly Supervised Semantic Segmentation**

The project is intentionally built for quick AutoDL pre-experiments. It freezes
CLIP, trains only a lightweight decoder, logs every run, and produces comparison,
ablation, and visualization artifacts that can be filled into the paper after
real experiments finish.

Important: the paper draft contains placeholders for experimental numbers. Do
not submit with invented results.

The bundled `paper/aaai2027.sty` is a compile-friendly placeholder. Replace it
with the official AAAI-27 author kit once AAAI releases the final style files.

## Files

- `paper/main.tex`: AAAI-style LaTeX draft.
- `paper/main.md`: Markdown preview of the same paper.
- `paper/references.bib`: BibTeX skeleton.
- `src/rpc_clip`: training, evaluation, model, metrics, visualization code.
- `configs/voc_rpc_clip.yaml`: default single-GPU VOC configuration.
- `experiments/*.csv`: result tables to fill after runs.
- `experiments/experiment_log.md`: human-readable experiment diary.
- `scripts/*.sh`: AutoDL setup, dataset download, training, packaging.
- `docs/autodl_zh.md`: Chinese AutoDL and Git runbook.

## Method Summary

RPC-CLIP keeps the CLIP ViT backbone frozen and trains a small patch decoder.
Training uses only image-level labels:

1. CLIP text-image patch similarity proposes foreground pseudo labels.
2. A momentum prototype bank accumulates class and background prototypes from
   high-confidence patches.
3. Text scores, prototype scores, and decoder predictions are fused with an
   entropy-based reliability weight.
4. A boundary-aware smoothness loss encourages coherent masks without using
   dense labels.

## Local Smoke Test

From this directory:

```bash
pip install -r requirements.txt
PYTHONPATH=src python train.py --config configs/voc_rpc_clip.yaml --smoke --output runs/smoke
```

The smoke mode uses random synthetic samples to verify code paths only. It does
not measure the method.

## AutoDL Quick Start

On an AutoDL single-GPU instance:

```bash
git clone <your-repo-url> aaai2027-rpc-clip
cd aaai2027-rpc-clip
bash scripts/autodl_setup.sh
bash scripts/download_voc.sh /root/autodl-tmp/datasets
bash scripts/run_voc_single_gpu.sh
```

After training:

```bash
bash scripts/package_results.sh runs/rpc_clip_voc
```

Send back the generated zip if results are weak; the run directory contains
config, logs, checkpoints, predictions, and visualizations.

For ablations:

```bash
bash scripts/run_ablations.sh
```

## Suggested Experiment Plan

Main comparison:

- VOC 2012 val: compare against CAM, SEAM, RRM, CLIP-ES, SFC, WeCLIP.
- VOC 2012 test: report only after the val pipeline is stable.
- Optional COCO pretrain/eval if compute budget allows.

Ablation:

- Text-only pseudo labels.
- Text + prototype calibration.
- Text + prototype + reliability weighting.
- Full RPC-CLIP with boundary-aware smoothness.
- Prompt template count and prototype momentum.

Visualization:

- Input image, pseudo mask, prediction, ground truth.
- Failure cases for small objects, multiple instances, confusing background.
- Reliability heatmaps if needed in the second iteration.
