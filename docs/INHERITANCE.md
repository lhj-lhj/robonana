# Inheritance and reuse map

## Runtime graph

```text
RoboTwinLeRobotDataset (initial FACT RoboTwin-v2)
  or RoboTwinHDF5Dataset (separate policy rollouts)
  -> FACT sampler + DefaultCollator + Trainer
    -> Flux2FACTModel (subclasses official Flux2)
      -> existing Flux2.double_blocks
      -> existing Flux2.single_blocks
      -> existing Flux2.final_layer for image output
      -> small action/state/value heads
      -> optional training-only DINO input/output heads
```

## Reused without copying

| Upstream | Reused code | robonana extension |
|---|---|---|
| FACT | sampler registry and `DefaultCollator` | LeRobot-v2 initial-data adapter and raw-HDF5 rollout adapter |
| FACT | `Trainer` loop, Accelerate/DeepSpeed, optimizer, checkpoint and logging | FLUX-specific forward/eval hooks |
| FLUX.2 | `Flux2`, image/text projections, RoPE and modulation | `Flux2FACTModel` subclass |
| FLUX.2 | all `DoubleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | all `SingleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | image `final_layer` | action/state/value linear heads |

## New trainable parameters

- shared action input projection for noisy and clean action tracks;
- shared state input projection for current and future state;
- value input projection;
- horizon embedding and segment embeddings;
- action, future-state, and value output heads.
- optional `Linear(3072, hidden_size)` / `Linear(hidden_size, 3072)` DINO heads.

There is no MoT, ActionDiT, or second transformer backbone.

## Token order

```text
[language | state | current image | noisy action | clean GT action |
 horizon | future state | value | future image VAE | future image DINO]
```

The attention mask is applied inside every reused FLUX.2 double-stream and single-stream block.
The noisy-action and full-clean-action tracks are independently causal. For a
sample with horizon `idx_h`, horizon/state/value/VAE/DINO target queries can read
only clean action tokens `G_1..G_idx_h`; they cannot read the clean action suffix
or any noisy-action token. This per-sample mask is rebuilt from the batch's
`idx_h` tensor on every forward.
The DINO suffix is a one-way auxiliary sink: it reads the complete world-model
prefix and itself, while every earlier token is blocked from reading DINO.
Inference omits this zero-length suffix and therefore keeps the existing action
and VAE-latent sampling path unchanged.

## Offline/online feature path

```text
RoboTwin instruction -> official FLUX.2 Qwen3Embedder.forward -> language_context.pt
RoboTwin HDF5 cameras -> FACT build_robotwin_three_view_tensor -> FLUX.2 AE -> frame tokens
RoboTwin three native camera frames at t_h -> frozen online DINOv3 ViT-B/16 -> 14x14x768 patches
  -> lossless 2x2 pixel-unshuffle per camera -> 3x(7x7x3072) frame features
```

The cache adds no alternative preprocessing geometry. It calls FACT's existing
three-view layout helper and reproduces the official FLUX.2 Klein VAE
patchify/BatchNorm sequence. One episode tensor is indexed as both
`current_latent[t]` and `future_latent[min(t + idx_h, T - 1)]`.
The DINO branch decodes only that same clamped future frame from each native
camera and computes `[147, 3072]` online. The frozen encoder is not an optimizer
parameter, is not written into RoboNana checkpoints, and is not loaded by
inference.

At `t_h = min(t + idx_h, T - 1)`, `future_state` is one state vector from exactly
that frame. The scalar value is the normalized time-to-go at `t_h`, not a single
label shared by every possible horizon or the final step of the action chunk.
