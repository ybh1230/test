# Team Publication Notes

These notes summarize the public directions found before drafting RPC-CLIP. They
are for positioning and writing style only; do not copy text from the papers.

## Closest Direction: Weakly Supervised Semantic Segmentation

- **Reliability Does Matter: An End-to-End Weakly Supervised Semantic
  Segmentation Approach**  
  Authors include Bingfeng Zhang and Jimin Xiao. The paper emphasizes reliable
  region mining from image-level labels and reports strong VOC results. RPC-CLIP
  borrows the high-level reliability theme but replaces CAM/FCN training with
  frozen-CLIP prompt-prototype calibration.

- **Affinity Attention Graph Neural Network for Weakly Supervised Semantic
  Segmentation**  
  Authors include Bingfeng Zhang and Jimin Xiao. The paper focuses on reliable
  propagation from confident seeds under bounding-box supervision. RPC-CLIP keeps
  the seed propagation intuition but avoids graph training for a faster single
  GPU pipeline.

- **SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation**  
  Authors include Xinqiao Zhao, Feilong Tang, Xiaoyang Wang, and Jimin Xiao. The
  paper uses class prototypes to calibrate shared features and improve CAM
  quality. RPC-CLIP turns this into CLIP prompt-prototype calibration.

- **Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation
  / WeCLIP**  
  Authors include Bingfeng Zhang, Siyue Yu, Yunchao Wei, Yao Zhao, and Jimin
  Xiao. This is the strongest local anchor for the current draft: frozen CLIP,
  lightweight decoder, dynamic pseudo-label refinement, and lower training cost.
  RPC-CLIP should be compared with WeCLIP and must provide a clear component
  difference through prototype reliability fusion.

## Related Team Style Signals

- **Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object
  Detection**  
  Uses a direct title, a simple conceptual principle, prototype generation, and
  self-contrastive learning. This supports a concise title and method framing
  around reliability/prototypes.

- **Discriminative Triad Matching and Reconstruction for Weakly Referring
  Expression Grounding**  
  The writing highlights a lightweight module and clear problem diagnosis. This
  supports keeping the method section compact and emphasizing why each component
  is necessary.

- **Fast Pixel-Matching for Video Object Segmentation**  
  Highlights speed-performance balance. RPC-CLIP should report trained
  parameters, GPU hours, and single-GPU setting.

- **IAN / segmentation-mask guided person search**  
  These papers consistently connect auxiliary dense cues to more discriminative
  recognition. RPC-CLIP can echo this by arguing that prototypes and reliability
  provide dense cues for a frozen vision-language model.

## Paper Positioning

Recommended claim shape:

1. Frozen CLIP is strong but raw patch-text pseudo labels are noisy.
2. Prior reliability mining and shared feature calibration show that trustworthy
   regions/prototypes matter.
3. RPC-CLIP introduces a lightweight prompt-prototype reliability loop that
   trains only a decoder and therefore remains practical on one GPU.
4. The paper must prove this with:
   - comparison against WeCLIP and CLIP-ES;
   - ablation of text-only, prototype, reliability, and boundary terms;
   - qualitative pseudo-mask and prediction visualizations;
   - training time and trainable parameter count.

