# Image input audit, 2026-09-08

## Historical training evidence

Read the saved Stage-1 config on 190, rather than inferring from the run name:
`experiments/hanging_mug_mac_world_5000_to_15000_merged100_plus_randomized550_20260906/config.json`.
Its original pool selects `Clean/hanging_mug` (50 metadata episodes) and
`Randomized/hanging_mug` (500), under
`/workspace/datasets/fact-robotwin-v2/RoboTwin`.
Replay is `/data3/hongjia/robonana_rollouts/hanging_mug_round0_plus_eval50_100_20260906`,
with 100 HDF5 episode files. All Stage-1 pools point to the FACT-v2 normalization
file. Pool weights are original .5, collected success .125, latest failure .375.

The old image pipelines were **not identical**:

- LeRobot MP4 used FACT's non-antialiased per-view resize and normalization.
- HDF5 replay used a separate antialiased resize.
- Both saved BF16 image features; live inference omitted that rounding.
- VAE native batches differed; historical caches do not certify the actual
  batch/backend/runtime. Cache CLI defaults are not evidence of actual runs.
- Replay saved lossy JPEG; raw online pixels and decoded replay pixels differ.

A previously checked replay frame had old HDF5/live normalized-pixel max
difference .37638580799 (mean .00177359162). That is preprocessing, not proof
of an attention-mask bug. Old cache metadata references a now-absent migration
VAE directory and has no weight hash: historical weight equality cannot be
certified from the directory name alone.

## Maintained replacement and validation

README documents the one maintained pipeline and `latents_v2` contract. Code
reuses FACT pixel helpers and the existing FLUX packing/BN implementation;
Qwen, MAC, action sampling, Q/V architecture and losses are not changed.
New PNG HDF5 RGB is lossless. Existing compressed data stays readable, but its
lost pixels cannot be recovered by converting it to PNG.

At commit `1c83eb0`, 190 ran 155 tests successfully (one skipped). A real VAE
probe on B200 (`cuda:6`) compared two frames per source: original Clean
episode 0 and replay episode 0. Cache generation/live B1/live B2 were bitwise
equal, maximum absolute latent error 0, shape `[2,288,128]`.
Runtime: torch 2.8.0+cu128, diffusers .36.0, torchvision .23.0+cu128, Pillow
11.3.0, av 12.3.0. Current VAE contract SHA256:
`1716c6fa967f91031e8016d04e4c5e7877add5c07bcfa2f0c1d505472b45dfca`.
The probe does not mutate datasets or their caches.

## Additional blocker: critic normalization

The saved `hanging_mug_critic_7k_to_17k_bs16_20260908/config.json` uses FACT-v2
stats for original data but `/workspace/datasets/RoboTwin/hf_dataset/robonana_norm_stats.json`
for replay. Their state/action mean/std differ, not merely their JSON metadata.
Current preflight rejects mixed values. This cannot be fixed retroactively
inside already-trained Q/V weights; choose consistent Stage-1 statistics and
retrain/evaluate critic under a new explicit run. Do not silently rewrite the
saved experiment or treat its metrics as a clean VAE-only ablation.

## Operational boundary

No full cache regeneration, checkpoint modification or training was started
for this audit. Old `latents` caches are retained but rejected by current
training. Explicitly regenerate image caches for both original and replay
pools before training. Existing checkpoints can be loaded, but learned the
historical mixed inputs; they are not certified as trained with the new
pipeline. Recheck action/Q evaluation after the transition.
