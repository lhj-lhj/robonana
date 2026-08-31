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
      -> small action/state/direct-reward/success/Q heads
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
| FLUX.2 | image `final_layer` | action/state/direct-reward/success/Q linear heads |

## New trainable parameters

- shared action input projection for noisy and clean action tracks;
- shared state input projection for current and future state;
- learned direct-reward and success query tokens plus a scalar Q input projection;
- horizon embedding and segment embeddings;
- action, future-state, direct-reward, success-logit, and Q output heads.
- optional `Linear(3072, hidden_size)` / `Linear(hidden_size, 3072)` DINO heads.

There is no MoT, ActionDiT, or second transformer backbone.

## Token order

```text
[language | state | current image | noisy action | clean GT action |
 horizon | future state | reward | success | Q | future image VAE | future image DINO]
```

The single-horizon order above is unchanged for training. Multi-horizon
Stage-2 inference extends only the suffix:

```text
[language | state | current image | A | G |
 H_1 | S_1 | R_1 | U_1 | Q_1 | I_1 | H_2 | S_2 | R_2 | U_2 | Q_2 | I_2 | ... |
 H_T | S_T | R_T | U_T | Q_T | I_T]
```

`A` is zero-length in Stage-2. Every horizon block reads the shared clean
condition and only `G_1..G_h`; attention between different horizon blocks is
blocked in both directions. Omitting image prediction sets every `I_h` to zero
tokens, so all state/reward/success/Q horizons share one forward without FLUX-latent or
VAE-decoder work. DINO is always omitted at inference.

The attention mask is applied inside every reused FLUX.2 double-stream and single-stream block.
The noisy-action track A is bidirectional for joint diffusion denoising, while
the full-clean-action track G is causal. For a sample with horizon `idx_h`,
horizon/state/reward/success/Q/VAE/DINO target queries can read
only clean action tokens `G_1..G_idx_h`; they cannot read the clean action suffix
or any noisy-action token. This per-sample mask is rebuilt from the batch's
`idx_h` tensor on every forward.
The DINO suffix is a one-way auxiliary sink: it reads the complete world-model
prefix and itself, while every earlier token is blocked from reading DINO.
Inference omits this zero-length suffix.

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
that frame. The model's reward is the direct `-1/0` reward at that state and the
success logit marks the same terminal event. The dataset separately preserves
the discounted cumulative reward over the first `delta` valid transitions for
TD targets. Q is the full successful-demonstration MC return from `t`. Q is a
raw scalar flow target; reward/success are direct heads. None uses the removed
time-to-go normalization.

## Iterative-posttraining reuse

Posttraining does not add a policy, target-Q network class, action expert, or a
second world backbone. `FullModelEMA` is an FP32, eval-only deep copy of the
same trainable `Flux2FACTModel`; the frozen Qwen, FLUX AE, and DINO encoder live
outside that model and therefore are not copied. Candidate action generation
and EMA action/Q evaluation call the same `sample_flux2_action` and
`sample_flux2_world` helpers used by inference.

```text
four separate dataset views
  -> RoboTwinPosttrainSampler (pool -> task -> episode -> frame)
  -> online shared FLUX: best-of-8 candidate actions
  -> EMA shared FLUX: future-state/Q flow, direct reward/success, and argmax Q
  -> online shared FLUX training:
       pred_action target = behavior on success, EMA-ranked pseudo on failure
       clean G/world/reward/current-Q condition = behavior action everywhere
  -> optimizer + scheduler
  -> FP32 full-model EMA Polyak update
```

Posttraining Q targets are computed outside the online graph from the real
future observation and EMA policy/Q. Only successful terminals stop bootstrap;
RoboTwin failure endings are time-limit truncations. The reset-pre final row is
kept in HDF5 with `transition_valid=false`, so it can be a bootstrap observation
but can never create a zero-length TD transition or cross-episode reset edge.
