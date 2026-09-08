# Maintainer handoff (2026-09-06)

RoboNana now has one supported path: `mac_mot_v2`, fixed `chunk_horizon=48`.
Do not add variable-horizon or legacy checkpoint branches back into the main
training/inference code.

Read `docs/CURRENT_CODE_MAP.md` for the maintained entry points and the
2026-09-08 cleanup boundary. Historical experiment reports are evidence, not
alternate architecture specifications. Normalization has exactly one source:
`robonana.normalization.A_STATS_PATH` (FACT-v2 Stage-1 statistics A). Do not
derive statistics from a replay root or reintroduce replay-fitted statistics.
New continuation configs correct all pools to A without editing saved runs;
this is an intentional correction, not exact historical reproduction.

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
return/loss math remain FP32. Qwen weights/precision/language caches remain
unchanged. The 2026-09-08 image-consistency request supersedes the earlier
freeze on VAE preprocessing: one FACT resize + single-image FP32 VAE + BF16
roundtrip pipeline now serves both cache generation and live input. Only
`flux_cache/latents_v2` with matching contracts is accepted. Historical image
caches must be explicitly rebuilt; do not overwrite/relabel them or claim old
checkpoints were trained on the new inputs. New HDF5 RGB is lossless PNG.
See README's Unified image pipeline section and the bounded real-VAE probe.
Running processes retain their loaded code until explicitly restarted.

Critic-only continuation uses the maintained
`robonana.configs.critic_continuation.config` entry point, not `_tmp` adapters.
Set `ROBONANA_RESUME_CHECKPOINT` to the complete source checkpoint directory,
`ROBONANA_RESUME_CONFIG` to its run's config JSON, `ROBONANA_PROJECT_DIR` to a
new output directory, and `ROBONANA_MAX_STEPS=10000` for a 5k-to-10k extension.
This restores Q/V, Value EMA, Adam and progress via FACT, and rebases the restored
zero LR onto the extended cosine schedule without advancing its step. Replay,
warmup and optimizer hyperparameters are preserved; execution uses FP32 and
batch 8 x 2 GPUs x accumulation 1. It is not bitwise equivalent to the historical
BF16-rollout / batch-4-accumulation-2 execution. Keep the source run untouched.

For an explicitly requested larger continuation batch, set
`ROBONANA_BATCH_SIZE_PER_GPU` (default 8). Accumulation remains 1 on GPUs 6,7;
batch 16 therefore means global batch 32, not 16. The adapter does not linearly
scale the optimizer LR with batch size. `ROBONANA_MAX_STEPS` is the absolute
endpoint: an additional 10k updates from checkpoint 7k means 17000. Test actual
forward/backward memory before accepting a larger batch; preserve the source
checkpoint and use a new project directory for the extended run.

The environment path samples 32 action candidates, computes the L/S/I prefix
once, scores each candidate with Q, and executes `argmax Q`. One selected
success trajectory is eligible for the next round's BC pool.

For success episodes, the final observation is an absorbing terminal state and
the remainder of the 48-step window is padded. Failure episodes are sampled
only when a complete 48-step chunk exists and are never padded.

## Validation

Collection infrastructure probes and their verified timing/data-equivalence
results are documented in `docs/COLLECTION_POOL.md`. The opt-in collector loads
the pinned official RoboTwin RLinf-support VectorEnv directly, reuses isolated
environment processes and distributes accepted seeds through an atomic FIFO.
On 2026-09-08, doubling workers from two to four on GPUs 6/7 did not improve
throughput for the same eight episodes (1433.123 versus 1462.043 seconds).
Inference batch remains one; no RoboNana algorithm or production collection
default changed. Do not confuse GPU utilization or free memory with a measured
hardware throughput ceiling, or merge benchmark data into replay automatically.

Run on 190 after syncing the commit:

```bash
cd /data3/hongjia/robonana
python -m pytest -q
python scripts/diagnostics/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <mac checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <mac checkpoint>/config.json --device cuda:0 --smoke-forward
```

The 5,000-step hanging-mug world pilot is a completed historical experiment;
inspect its saved status/checkpoint before starting another run.
