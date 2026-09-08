# RLinf VectorEnv collection probe

This opt-in collector changes infrastructure only. It directly loads the official
`VectorEnv.step/reset/close/transform` methods, not a copied implementation:

https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/robotwin/envs/vector_env.py

The small `RoboNanaSubEnv` bridge follows that file's `SubEnv.reset` lifecycle
(persistent task, selective reset, clear caches every eight completed episodes).
It replaces the stock `SubEnv.step` because `gen_sparse_reward_data` changes our
per-control-step observation/label contract. Only the existing RoboNana adapter
executes policy actions and records transitions. No RLinf reward/PPO/GRPO code is
used. Reward fields in the vector transport are `None` and discarded; dataset
labels still come exclusively from the existing writer/training data pipeline.

## Isolation and invariants

- One GPU-bound process owns one native VectorEnv slot and one client. Multiple
  processes share the existing inference service. This avoids process-global RNG
  and render-device interference between different scenes in a shared interpreter.
- Do not add the dependency checkout to PYTHONPATH. Only its `vector_env.py` is
  imported by file; `envs.*` resolve to the CURRENT deployed RoboTwin task code.
- VectorEnv task/config construction is bypassed to inject the already-configured
  current task. No switch of robot, planner, assets, image quality or task branch.
- Fixed 48 actions, 20 denoising steps, M=32 argmax, candidate batch=16, FP32:
  unchanged. Server inference batch stays one even with multiple clients.
- Every control-step observation is recorded. Publish final observation before
  reset. Failed episodes remain real failures, not short successful fragments.
- Use prevalidated source seeds from the same task/config/assets and their exact
  recorded instructions. Never replace an invalid seed silently. This benchmark
  intentionally excludes expert seed discovery; count that cost separately when
  collecting previously unchecked seeds.
- New output directories only, atomic FIFO seed claims, bounded process supervision,
  source/commit config, GPU samples and final ledger/HDF5 consistency checks.
- Generated datasets are separate from training replay. No automatic training,
  merge, preprocessing or production-default change.

## Dependency on 190

Clone official source through Git (not patch deployment):

```bash
git clone --depth 1 --filter=blob:none --sparse --branch RLinf_support \
  https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin_RLinf
git -C third_party/RoboTwin_RLinf sparse-checkout set --no-cone \
  /robotwin/envs/vector_env.py /LICENSE
```

Required commit is `0008ae6800df9f75fc8de7098bacb01735fd8fd2`. If the branch has
moved, fetch and checkout that commit before running. Loader verifies HEAD and
the exact file bytes against Git. The dependency remains unmodified/untracked.

## Benchmark

`scripts/benchmark_robotwin_collection_pool.py --help` lists required inputs.
Use the model Python and existing RoboTwin Python separately. Source episode
HDF5s provide accepted seeds/instructions; they are not replayed action inputs.
The actual current policy chooses fresh actions using its existing stable seed.

- Serial control: `--sim-gpus 7 --server-gpu 6`.
- Parallel probe: `--sim-gpus 6 7 --server-gpu 6`.
- Four-environment probe: `--sim-gpus 6 7 6 7 --server-gpu 6`.

Workers now claim the next pending episode from `episode_queue.sqlite` whenever
their current episode finishes. The same prevalidated seed list is visible to all
workers; transactional ownership prevents duplicates. Claimed jobs are not silently
requeued after a crash: a failed worker aborts the probe, leaving auditable state.
All queue rows must be `done` and match the HDF5/ledger seed multiset before success.
Repeated GPU IDs mean separate persistent processes, not shared-thread scenes.

Use the identical source-episode list, checkpoint/config and different output
directories. Keep failed episodes in the comparison. Report episode wall time,
startup-inclusive total time, GPU memory, per-seed outcomes and frame counts.
Persistent scenes require real multi-episode stress testing before deployment.

## Measured result on 190 (2026-09-08)

Validated implementation commit: `7d2a1b0`. Runtime used the existing
`/data3/hongjia/venvs/robotwin-sapien303/bin/python`, invoked through its venv
path, NOT its resolved symlink target. The first attempt with the wrong resolved
interpreter was stopped, excluded from comparisons, and left separate on disk.

Final output:
`/data3/hongjia/robonana/outputs/collection_pool_parallel4_venv_20260908`.
Checkpoint: the existing critic step11000 under
`experiments/hanging_mug_critic_7k_to_17k_bs16_20260908`.

| Seed | GPU | Result | Actions | This worker's episode seconds |
|---|---:|---|---:|---:|
| 100000 | 6 | success | 328 | 207.738 |
| 100003 | 6 | success | 329 | 165.807 |
| 100001 | 7 | success | 326 | 200.168 |
| 100004 | 7 | failure | 900 | 374.529 |

Both workers reused the same process/task/client for two episodes. Total wall
time, including server/client startup and worker completion: **590.720 seconds**
(9m51s), 24.377 episodes/hour. Final server teardown is outside that timer.
The previous original-collector run of these same four seeds took **929.724
seconds** between its `.started` and `summary.txt` markers (15m30s).
Observed combined throughput ratio: **1.574x**, wall time reduction **36.5%**.
These are separate measurements of the combined infrastructure change, not a
fresh randomized A/B isolating parallelism alone. New seed-discovery costs are
not included in the new path, and timing markers are not identical instrumentation.

Peak sampled total GPU memory: GPU6 **41,577 MiB**, GPU7 **7,037 MiB**.
This is a small 75%-success sample with unequal worker loads; do not advertise
its linear 100-episode extrapolation as a validated production collection time.

All four output HDF5s were compared against the original collection by seed:

- 1,887 observations, all three JPEG streams byte-identical (5,661 images).
- All `joint_action/vector` and `policy_action/vector` arrays exactly equal.
- All candidate Q arrays, selected Q/index, Q margins and candidate counts equal.
- All `transition_valid` arrays equal; true terminal observation retained.
- Success/failure and trajectory lengths identical, including the 900-step fail.
- Supervisor's unique-seed, frame-count and final-observation checks passed.

Eight unit tests passed locally and on 190. No models, losses, inference server,
sampling, dataset writer or training files changed. This verifies two resets per
worker, not a 50/100-episode stability run or real eight-reset cache-clear cycle.
Production collection defaults remain unchanged until a longer stress test.

The old render-sync probe and its test/report were removed at the user's request;
their historical results remain recoverable in Git. Its ~1% simulated-step gain
did not justify enabling it. Remote raw benchmark artifacts were not removed.
