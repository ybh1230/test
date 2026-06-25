# Experiment Log

Fill this file immediately after each run. Keep failed runs: they are useful for
debugging the method.

## Run Template

- Date:
- Machine/GPU:
- Git commit:
- Config:
- Dataset path:
- Command:
- Training time:
- Best checkpoint:
- Best val mIoU:
- Test mIoU:
- Notes:
- Failure cases:
- Next change:

## RPC-CLIP Prototype Runs

### v1/v2

- Observation: val mIoU stayed around 2.16%; background IoU was 0.
- Diagnosis: pseudo labels collapsed toward foreground.

### v3

- Observation: val mIoU improved only to about 2.95%.
- Diagnosis: pseudo labels had around 97.6% valid pixels and around 80% foreground,
  so supervision was too dense and noisy.

### v4/v6

- Observation: conservative quantile seeds produced healthier pseudo labels,
  with about 76% valid pixels and about 76% background among valid seeds, but
  val mIoU stayed around 3.9%.
- Diagnosis: the small decoder/frozen patch feature route is too weak as a main
  AAAI submission path.

### Pivot

- Decision: use WeCLIP/Frozen CLIP as the reproducible strong baseline and add
  reliability/prototype calibration on top.
