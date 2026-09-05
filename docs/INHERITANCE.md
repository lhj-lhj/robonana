# Inheritance and reuse map

## Runtime graph

```text
RoboTwinLeRobotDataset (initial FACT RoboTwin-v2)
  or RoboTwinHDF5Dataset (separate policy rollouts)
  -> FACT sampler + DefaultCollator + Trainer
    -> Flux2FACTModel or MacFlux2FACTModel (subclasses official Flux2)
      -> existing Flux2.double_blocks
      -> existing Flux2.single_blocks
      -> existing Flux2.final_layer for image output
      -> small action/state/Value/reward/success/Q heads
```

## Reused without copying

| Upstream | Reused code | RoboNana extension |
|---|---|---|
| FACT | sampler registry and `DefaultCollator` | LeRobot-v2 initial-data adapter and raw-HDF5 rollout adapter |
| FACT | `Trainer` loop, Accelerate/DeepSpeed, optimizer, checkpoint and logging | FLUX-specific forward/eval hooks |
| FLUX.2 | `Flux2`, image/text projections, RoPE and modulation | `Flux2FACTModel` / `MacFlux2FACTModel` subclasses |
| FLUX.2 | all `DoubleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | all `SingleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | image `final_layer` | action/state/Value/reward/success/Q heads |

The adapters add action/state projections, learned query and segment tokens,
and small output heads. There is no MoT, ActionDiT, or second transformer
backbone.

## Current fixed-chunk MAC token order

```text
[language | state | current image VAE | Value | predicted action |
 clean action chunk | Q | reward | success | future state | future image VAE]
```

`mac_v1` has one fixed 48-step chunk and no `idx_h` token. The maintained MAC
training config disables DINO. The model implementation can append an optional
training-only DINO sink after `future image VAE`, but no earlier output may read
it.

The mask is applied in every reused FLUX.2 double-stream and single-stream
block. With `C = [language, state, current image]`, `A = predicted action`, and
`G = clean action chunk`, the allowed dependencies are:

| query | readable keys |
|---|---|
| `Value` | `C` and its own query token |
| `A` | `C` and all 48 `A` tokens |
| `G` | `C` and all 48 `G` tokens |
| `Q` | `C`, all 48 `G` tokens, and its own query token |
| `reward` | `C`, all 48 `G` tokens, and reward |
| `success` | `C`, all 48 `G` tokens, reward, and success |
| `future state` | `C`, all 48 `G` tokens, reward, success, and future state |
| `future image` | `C`, all 48 `G` tokens, reward, success, future state, and future image |

Reward, success, future state, and future image cannot read Value, predicted
action, or Q. Q cannot read reward, success, or either future. Value cannot
read either action track. Value and Q are deterministic scalar heads; they are
not flow-corrupted or sampled.

## Legacy checkpoint token order

Old `legacy_v1` checkpoints retain their variable-horizon layout for strict
loading and historical inference:

```text
[language | state | current image | A | G |
 H | S | R | U | Q | I | optional DINO]
```

Those checkpoints may pack isolated horizon blocks and use `G_1..G_idx_h`.
They must never be silently interpreted as `mac_v1`.

## Offline/online feature path

```text
RoboTwin instruction -> official FLUX.2 Qwen3Embedder.forward -> language_context.pt
RoboTwin HDF5 cameras -> FACT build_robotwin_three_view_tensor -> FLUX.2 AE -> frame tokens
```

The cache adds no alternative preprocessing geometry. It calls FACT's existing
three-view layout helper and reproduces the official FLUX.2 Klein VAE
patchify/BatchNorm sequence. For MAC, one episode tensor is indexed as both
`current_latent[t]` and `future_latent[min(t + 48, T - 1)]`; `future_state`
comes from that same endpoint.

One reward query emits 48 Bernoulli logits. Valid non-goal steps are class 0
(`-1`), the absorbing suffix after a successful terminal is class 1 (`0`), and
clipped timeout padding is masked out. The success query predicts termination
at the chunk endpoint. The scalar chunk return is derived from these 48 logits;
it is not directly regressed.

## MAC posttraining reuse

MAC does not add a policy, target-Q class, action expert, or second world
backbone. `FullModelEMA` is an always-on FP32, eval-only deep copy of the same
trainable `MacFlux2FACTModel`; frozen Qwen and FLUX AE modules live outside it.
Candidate generation and Q evaluation use the same samplers as environment
inference.

```text
four separate dataset views, fixed future = t + 48
  -> online shared FLUX trains real image/state/reward/success targets
  -> action BC is enabled only on successful trajectories
  -> sample M=8 independent action chunks from the online BC flow
  -> deterministic online Q selects the argmax candidate
  -> learned world model rolls that selected chunk forward once (H=1)
  -> EMA Value bootstraps the detached imagined return
  -> deterministic online Value and Q regress that return
  -> optimizer + scheduler -> FP32 full-model EMA Polyak update
```

At environment inference, M=32 independent BC chunks are ranked by the same
deterministic Q and only the argmax chunk is executed. The rollout writer stores
candidate Q values and the selected index. On the next training round, selected
successful trajectories enter action BC; selected failures train the learned
world/reward/success targets but have zero action-BC weight. The reset-pre final
row remains `transition_valid=false` and cannot create a reset edge.

## 120k migration boundary

`load_mac_from_legacy_checkpoint` reconstructs the immutable source from its
saved `config.json`. It whitelists the official FLUX.2 backbone plus compatible
action/state/image adapters. Legacy horizon/segment/DINO/Value and other project
heads are skipped even if a tensor shape happens to match; new MAC query tokens
and heads are initialized explicitly. Once a `mac_v1` checkpoint exists, later
rounds must use exact `trained` loading with that run's `config.json`.
