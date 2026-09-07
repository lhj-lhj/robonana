# RoboNana — `mac_mot_v2`

This repository maintains one RoboTwin training and inference path: a fixed action chunk of **48 steps**, one FLUX backbone, a deterministic online Q expert, and a deterministic Value expert with an EMA target copy. The implementation follows the public [MAC repository](https://github.com/kwanyoungpark/MAC) and the cached MoT expert pattern from [ImageWAM](https://github.com/yuyangalin/ImageWAM).

The old `idx_h`/variable-horizon, 800M, full-FLUX-EMA, TD/MC, and 120k runtime-loading paths are removed. The original 120k checkpoint is an external archived artifact and is not deleted; it is no longer a valid runtime input. Every new run starts from the current 1,000-step MAC checkpoint unless `ROBONANA_MAC_PRETRAIN_CHECKPOINT` explicitly points to another complete `mac_mot_v2` checkpoint.

## Current architecture

The trainable model is `MacFlux2FACTModel`:

* Phase 1 (`world_policy`): FLUX actor/world parameters train. Successful windows train action BC; successful and failed windows train the world targets.
* Phase 2 (`critic`): the complete FLUX backbone is frozen and only `value_expert` and `q_expert` train. Value has a float32 EMA target; Q has no EMA.
* Both experts read the frozen FLUX per-layer K/V through the ImageWAM-style MoT adapter. Scalar queries and scalar heads are newly initialized; FLUX blocks/modulation are copied or scaled.
* Live inference always samples an action chunk and performs Q rejection sampling (`M=32` by default). The L/S/I prefix is computed once and reused for every candidate.

The maintained sequence is:

```text
[language | state | current_image | pred_action | clean_action_chunk |
 reward[48] | success | future_state | future_image]
```

The information boundaries are strict: success sees language/state/current-image/clean-action/reward; future state additionally sees success; future image additionally sees future state. Q sees only language/state/current-image/clean-action. Value sees only language/state/current-image.

## MAC losses

For a real or imagined chunk of 48 actions, `r_t` is the per-step binary reward logit (class 0 decodes to `-1`, class 1 to `0`). The Value target is the fixed-horizon n-step target used by MAC:

\[
L^V(\phi)=\mathbb E\left[\left(V_\phi(\hat s_{t+kn})-\sum_{i=k}^{H-1}\gamma^{(i-k)n}\hat r_{t+in}-\gamma^{(H-k)n}V_{\bar\phi}(\hat s_{t+Hn})\right)^2\right].
\]

With one imagined 48-step chunk (`n=48`, `H=1`) this reduces to the terminal chunk return plus the EMA Value bootstrap. The Q regression is:

\[
L^Q(\phi)=\mathbb E\left[(Q_\phi(s_t,\hat a_t)-\hat r_t-\gamma^{48}V_{\bar\phi}(\hat s_{t+48}))^2\right].
\]

Q is trained from learned-world-model imagined rollouts, not a fabricated real-replay Q label. Value and Q are deterministic functions at evaluation time.

## Training loop

1. Load the 1,000-step checkpoint and collect/prepare fixed-48 replay windows.
2. Phase 1 trains the action/world model. BC loss is multiplied by `action_loss_mask`, which is one only for successful trajectories. World losses always run on both success and failure data.
3. For successful terminal windows, observations after the terminal frame are padded as an absorbing state for the remaining chunk suffix. Failure windows are never padded: their final complete chunk ends exactly at the last recorded observation.
4. Phase 2 freezes FLUX, creates/loads the online Value and Q experts from the current Phase-1 checkpoint, and initializes the Value EMA by copying the **online Value expert** (never the previous EMA). The EMA is updated after each optimizer step. Q has no target copy.
5. For each critic batch, the world model performs one on-policy imaginary rollout. Candidate actions are sampled from the action flow, scored by Q, and the argmax candidate is selected. Successful selected environment trajectories are appended to the replay pool and later provide BC updates in the next Phase-1 round.

The default pilot configuration uses one imagined chunk, eight critic candidates during training, 32 candidates in live environment inference, and 20 flow steps. These are configuration knobs, not alternate algorithms.

## Data contract

The canonical dataset is `RoboTwinHDF5Dataset` (the LeRobot adapter reuses the same fixed-48 contract). Each sample contains:

`context`, `current_latents`, `future_latents`, `state`, `future_state`, `behavior_action`, `action_valid_mask`, `reward_chunk`, `reward_chunk_mask`, `success`, `reward`, `reward_h`, and `chunk_horizon=48`.

There is no sampled `horizon_idx`. Dataset windows are success starts `0..T-2` with absorbing suffix padding, or failure starts `0..T-49` with no padding. A failure episode shorter than one complete 48-step chunk is excluded.

## Checkpoints and commands

The default source is:

```text
/data3/hongjia/robonana/experiments/hanging_mug_mac_pilot_20260906/world_policy/
  models/checkpoint_epoch_1_step_1000/transformer/diffusion_pytorch_model.bin
```

Run from the repository root:

```bash
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_4b_mac.config
```

For a new collection/training round:

```bash
bash scripts/run_hanging_mug_mac_round.sh
```

Validate a complete checkpoint:

```bash
python scripts/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <checkpoint>/config.json --smoke-forward
```

The loader is intentionally strict: architecture must be `mac_mot_v2`, chunk/reward dimensions must be 48, and all non-expert FLUX parameters must be present. Frozen FLUX parameters are included in the checkpoint-resume protocol so Phase 2 can resume safely under DeepSpeed.

## Repository map

* `src/robonana/models/mac_flux2_fact.py` — fixed-48 FLUX actor/world model.
* `src/robonana/models/flux2_scalar_expert.py` — ImageWAM-style deterministic MoT scalar expert.
* `src/robonana/models/pretrained.py` — strict MAC checkpoint loading and trainable-surface control.
* `src/robonana/data/robotwin_hdf5.py` — success/failure windowing and absorbing-terminal targets.
* `src/robonana/sampling.py` — action flow, one-chunk world rollout, and Q rejection sampling.
* `src/robonana/training/robotwin_trainer.py` — two-phase training and Value EMA updates.
* `scripts/start_mac_world_pilot.py` — bounded world-model pilot and probes.

## Configuration reference

The canonical entry point is `robonana.configs.robotwin_flux2_4b_mac.config`.
It imports the common FACT/FLUX dimensions and applies the MAC overlay:

| Setting | Current value |
|---|---:|
| architecture | `mac_mot_v2` |
| action/state dimensions | 14 / 14 |
| action chunk and maximum horizon | 48 / 48 |
| reward head | `binary_chunk`, 48 logits |
| Value/Q output | one scalar each |
| critic candidates (training) | 8 |
| candidates (environment) | 32 |
| imagined chunks per critic batch | 1 |
| Value EMA decay | 0.995 |
| flow sampling steps | 20 |
| default dtype | BF16 FLUX, FP32 EMA storage |
| new trajectories per collection round | 100 total, successes and failures |
| stage 1 world/policy budget per round | 20,000 optimizer steps |
| stage 2 Value/Q budget per round | 10,000 optimizer steps |

The standard round is: collect 100 new trajectories with the current Q-selected
policy, prepare/cache them and mix with existing replay, train stage 1 for 20k
steps, then freeze FLUX and train stage 2 for 10k steps. These are additional
per-phase budgets, not lifetime checkpoint step numbers. Each new phase uses
a matching learning-rate decay length. Carry forward online model weights;
initialize Value EMA from online Value at the start of the new critic phase.

`run_hanging_mug_mac_round.sh` consumes the replay already collected for round r,
trains these two phases, and collects the next 100 trajectories for round r+1.
`ROBONANA_MAC_COLLECTION_EPISODES`, `ROBONANA_MAC_WORLD_POLICY_STEPS`, and
`ROBONANA_MAC_CRITIC_STEPS` override these defaults. `ROBONANA_MAX_STEPS` overrides
the phase budget when loading the canonical training config directly. Evaluation
episode counts are independent of collection counts. The explicitly named
historical pilot retains its small experimental budgets.

A critic-only control is an optional diagnostic: keep FLUX and replay fixed and
continue only Q/Value training, then compare evaluation results. It isolates the
effect of extra critic updates; it is not an extra stage in the standard round.

Useful overrides are `ROBONANA_MAC_SOURCE_RUN`,
`ROBONANA_MAC_PRETRAIN_CHECKPOINT`, `ROBONANA_MAC_PRETRAIN_CONFIG`,
`ROBONANA_MAC_PHASE` (`world_policy` or `critic`),
`ROBONANA_COLLECTION_ROUND`, `ROBONANA_REPLAY_ROOT`,
`ROBONANA_MAC_TRAIN_CANDIDATES`, `ROBONANA_MAC_EVAL_CANDIDATES`, and
`ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE`. The source checkpoint and saved
`config.json` must describe the same complete MAC model.

## Exact phase behavior

### Phase 1 — world/policy

Each minibatch mixes original successful demonstrations, selected-policy
successes, historical failures, and latest failures. An empty optional pool is
redistributed by the sampler instead of failing a round.

For clean action `G` and noisy action `A`, FLUX predicts the action flow target
and, from pure noise, the future image and future state. The 48 reward logits
use this target convention:

```text
real transition before terminal:  class 0 -> reward -1
absorbing terminal suffix:        class 1 -> reward  0
failed time-limit suffix:         invalid mask (not padded)
```

`action_loss_mask=1` only for successful episodes. `reward_chunk_mask` masks
unknown failed tails. Failures therefore train the world model without being
treated as demonstrations; there is no full-model EMA.

### Phase 2 — deterministic critics

The trainer exact-loads the preceding Phase-1 online checkpoint, freezes every
FLUX/world parameter, and enables gradients only for `value_expert.*` and
`q_expert.*`.

At the beginning of a new critic phase:

```text
online Value  <- exact Value expert from loaded checkpoint
online Q      <- exact Q expert from loaded checkpoint
target Value  <- deepcopy(online Value), converted to FP32
```

When resuming the same critic phase, the saved target Value and EMA metadata
are restored. A new phase never initializes from the previous phase's target
Value. After every finite optimizer step, target Value is Polyak-averaged; Q
is never EMA-updated.

Each critic batch performs exactly one learned 48-step imaginary transition.
Candidates are generated by the same action flow sampler used online, Q picks
the best candidate, and that selected action is fed to the world model. With
`R_chunk` decoded from the 48 binary reward logits:

```text
terminal = (sigmoid(success_logit) >= 0.5)
V_target = R_chunk + gamma**48 * (1-terminal) * target_V(next_state)
Q_target = R_chunk + gamma**48 * (1-terminal) * online_V(next_state)
```

Both targets are stop-gradient. Any non-finite loss or gradient flag is
reduced across ranks with a true minimum/AND check; all ranks skip the update
before backward, optimizer, scheduler, or EMA work.

## Q rejection and prefix reuse

For one request, the frozen FLUX prefix is computed once and shared:

```text
C = FLUX([language, state, current image])
for candidate m in groups:
    A_m = action-flow(C, noise_m)
    Q_m = Q-expert(C, clean_action=A_m)
return action[argmax_m Q_m]
```

L/S/I can attend only to L/S/I; each candidate's clean-action tokens can
attend to C and itself; candidates cannot read one another. Candidate groups
bound peak memory while preserving the full M-way result. The benchmark
script compares cached and uncached M=1/8/32 execution.

## Replay collection and selected-policy BC

`scripts/collect_prepare_robotwin_rollouts.sh` runs isolated RoboTwin
evaluation with the current checkpoint, writes a separate rollout root, and
then caches FLUX language/image latents. Metadata records checkpoint, policy
version, round, success, terminal observation, and time-limit truncation.
The next round mixes these records through the four configured pools.

Only Q-selected successful trajectories contribute action BC. Failed
trajectories remain replay data and contribute world-model supervision. The
collection script is resumable by its episode ledger; never merge its output
into the original demonstration tree manually.

## Validation checklist on 190

Before a formal run, record commit, saved config, checkpoint, dataset index,
process command, logs, and GPU ownership. Then run:

```bash
cd /data3/hongjia/robonana
git status --short
nvidia-smi
python -m pytest -q
python scripts/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <checkpoint>/config.json --device cuda:0 --smoke-forward
```

For a world-only pilot, use `scripts/start_mac_world_pilot.py`. It writes a
source manifest, probes fixed windows before and after training, and stops at
`world_complete_review_required`; inspect metrics before starting critic. The
completed hanging-mug 5,000-step pilot remains in its existing experiment
directory and must not be overwritten.

Recommended round order:

```text
1. load the 1,000-step (or previous exact MAC) checkpoint
2. collect selected-policy success/failure rollouts
3. prepare/cache replay data
4. run world_policy
5. inspect fixed-window world fit and finite-loss logs
6. run critic with frozen FLUX and Value-only EMA
7. evaluate M=1 versus M=32 in RoboTwin
```

## Checkpoint and resume protocol

DeepSpeed exports complete FLUX weights even when FLUX is frozen in critic.
Expert and EMA files are stored alongside the transformer export. A checkpoint
is accepted only when architecture metadata, dimensions, reward-head type, and
saved model config agree with `mac_mot_v2`.

If resume reports missing frozen FLUX keys, it is an incomplete pre-fix
checkpoint and must not be force-loaded. Start from the nearest complete MAC
checkpoint. If only target Value is missing at a new critic phase, copy the
loaded online Value expert into a fresh FP32 target; do not copy an unrelated
target from another run.

## Numerical policy

BF16 is used for FLUX activations and expert forwards on supported GPUs. Value
EMA storage and updates are FP32. Inference sanitizes decoded actions with a
finite fallback and clips to normalization bounds. When diagnosing instability,
first lower critic learning rate or candidate count; do not silently add a
second EMA or target Q.

## Archived history

The original 120k checkpoint remains outside the maintained runtime path for reproducibility. Its conversion was a one-time preprocessing operation; no legacy architecture, loader, 800M configuration, or variable-horizon training mode is kept in this repository.
