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
- New output directories only, disjoint seed shards, bounded process supervision,
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

Use the identical source-episode list, checkpoint/config and different output
directories. Keep failed episodes in the comparison. Report episode wall time,
startup-inclusive total time, GPU memory, per-seed outcomes and frame counts.
Persistent scenes require real multi-episode stress testing before deployment.

The old render-sync probe and its test/report were removed at the user's request;
their historical results remain recoverable in Git. Its ~1% simulated-step gain
did not justify enabling it. Remote raw benchmark artifacts were not removed.
