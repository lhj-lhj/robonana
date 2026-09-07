# Maintainer handoff (2026-09-06)

RoboNana now has one supported path: `mac_mot_v2`, fixed `chunk_horizon=48`.
Do not add variable-horizon or legacy checkpoint branches back into the main
training/inference code.

## Operational boundaries

* Source checkout: `D:\Robotic\robonana`
* GitHub: `https://github.com/lhj-lhj/robonana` (`main`)
* Validation host: `hongjia@208.64.254.190`
* Server checkout: `/data3/hongjia/robonana`
* Default initialization: the 1,000-step checkpoint under
  `/data3/hongjia/robonana/experiments/hanging_mug_mac_pilot_20260906/world_policy`

Never delete or overwrite server datasets, replay pools, experiment outputs,
logs, or checkpoints. The archived 120k checkpoint remains available as a
user artifact but is not loaded by runtime code.

### Source synchronization (2026-09-07)

GitHub `main` is the single source of truth for maintained code. Validate and
commit locally, push to `origin/main`, then update 190 with `git pull --ff-only`.
Verify `git rev-parse HEAD` matches on both machines and GitHub before launching
a new experiment. Do not deploy source with patch files, SCP overlays, or a
parallel validation checkout. The tracked SAPIEN dependency patch is an upstream
dependency fix, not a source-deployment mechanism.

If 190 has uncommitted source changes, inspect and preserve them first; do not
force-reset or overwrite them. Integrate any needed changes into GitHub before
updating. Runtime directories (`_tmp/`, `experiments/`, `eval/`, checkpoints,
datasets, outputs and credentials) are not synchronized through Git. Existing
processes retain loaded code; a checkout update is not a process restart or a
guarantee that an already running experiment uses the new commit.

## Model and optimization contract

The single FLUX sequence is

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward[48] | success | future_state | future_image_vae]
```

Phase 1 trains FLUX actor/world parameters. BC is success-masked; world losses
use both success and failure windows. Phase 2 freezes FLUX and trains only the
deterministic Value/Q MoT experts. Value has one FP32 EMA target; Q has no EMA.
The Value EMA is initialized from the current online Value expert at the start
of every new critic phase and restored only when resuming that same phase.

Standard new-round budgets (updated 2026-09-07): collect 100 new trajectories
total (successes plus failures), train phase 1 for 20,000 optimizer steps, then
phase 2 for 10,000 optimizer steps. Each new phase's LR decay matches its budget.
The round launcher starts from already collected replay and collects 100 more
episodes for the following round after training. A critic-only continuation is
an optional diagnostic, not part of this default cycle. Do not reinterpret
these defaults as permission to launch training or modify a saved ongoing run.

New Stage-1/Stage-2 runs default to GPUs 6,7, batch 8 per GPU and gradient
accumulation 1 (effective batch 16). Explicit environment overrides remain
supported. Existing saved experiment configs and running processes are not
rewritten; a historical continuation can still restore batch 4 / accumulation 2.

RoboNana is FP32-only: FLUX training, action/world rollout, online/target Q/V
and environment policy inference. No precision selector is maintained; non-FP32
training overrides/checkpoint-load requests fail early. EMA storage/update and
return/loss math remain FP32. Frozen external encoders (Qwen/VAE), their cache
generation and existing cache storage must remain unchanged per user request.
Their features are cast to FP32 at the RoboNana model input boundary.
The active critic-only process was not restarted and retains its loaded code.

The environment path samples 32 action candidates, computes the L/S/I prefix
once, scores each candidate with Q, and executes `argmax Q`. One selected
success trajectory is eligible for the next round's BC pool.

For success episodes, the final observation is an absorbing terminal state and
the remainder of the 48-step window is padded. Failure episodes are sampled
only when a complete 48-step chunk exists and are never padded.

## Validation

Run on 190 after syncing the commit:

```bash
cd /data3/hongjia/robonana
python -m pytest -q
python scripts/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <mac checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <mac checkpoint>/config.json --device cuda:0 --smoke-forward
```

The 5,000-step hanging-mug world pilot is a completed historical experiment;
inspect its saved status/checkpoint before starting another run.
