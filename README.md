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

## Archived history

The original 120k checkpoint remains outside the maintained runtime path for reproducibility. Its conversion was a one-time preprocessing operation; no legacy architecture, loader, 800M configuration, or variable-horizon training mode is kept in this repository.
