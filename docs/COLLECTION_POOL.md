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
- Fixed 48 actions, 20 denoising steps, M=32 argmax and FP32: unchanged.
  Candidate batch defaults to 16 and request batch to one; explicit batching
  probes may change grouping, not the candidate count or sampling algorithm.
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

`scripts/diagnostics/benchmark_robotwin_collection_pool.py --help` lists required inputs.
Use the model Python and existing RoboTwin Python separately. Source episode
HDF5s provide accepted seeds/instructions; they are not replayed action inputs.
The actual current policy chooses fresh actions using its existing stable seed.

- Serial control: `--sim-gpus 7 --server-gpu 6`.
- Parallel probe: `--sim-gpus 6 7 --server-gpu 6`.
- Four-environment probe: `--sim-gpus 6 7 6 7 --server-gpu 6`.
- Two-request / 32-candidate grouping: add `--inference-batch-size 2
  --candidate-batch-size 32 --batch-wait-ms 10`. Incomplete batches flush after
  the bounded wait; `batch_metrics.jsonl` records actual batch sizes and timings.
  Four workers can feed this two-request service without requiring a global
  environment step barrier. Production defaults remain unchanged.

Before a changed grouping is used for collection, the offline
`scripts/diagnostics/benchmark_robotwin_inference_batch.py` probe compares warmed 1x16,
1x32 and 2x32 inference on identical recorded images/state/instructions and
fixed probe noise seeds. It reports all-candidate action/Q differences, argmax
indices and CUDA allocated/reserved peaks, and refuses a silent argmax change.
This does not guarantee bitwise equality on all future inputs: batched kernels
can differ numerically even in FP32, including the unchanged frozen encoders.

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

## Dynamic queue: two versus four environments (2026-09-08)

Both runs used implementation commit `5dd6730`, the same step11000 checkpoint,
the same eight accepted seeds/instructions, and only B200 GPUs 6/7. Both kept
FP32, 48 actions, 20 denoising steps, M=32, candidate batch=16 and server request
batch=1. The only A/B difference was worker count/placement; both used the new
FIFO queue. The two-environment run finished before the four-environment run
started. These are one trial per configuration, not repeated confidence bounds.

| Configuration | Worker GPU slots | Wall time, startup included | Episodes/hour | Peak GPU6 / GPU7 memory |
|---|---|---:|---:|---:|
| 2 persistent environments | 6, 7 | 1433.123 s (23m53s) | 20.096 | 41,577 / 7,037 MiB |
| 4 persistent environments | 6, 7, 6, 7 | 1462.043 s (24m22s) | 19.698 | 48,605 / 14,065 MiB |

Four environments took **2.02% longer**, with **0.980x throughput**. This small
difference does not establish a statistically meaningful slowdown, but it shows
no useful gain from doubling environments in this placement. Keep two workers
as the measured baseline; do not call this the B200 hardware ceiling.

Mean sampled GPU utilization after the first 60 seconds (including queue drain)
was GPU6/GPU7 **81.83%/46.03%** for two environments and **97.87%/36.48%** for four.
Samples were taken approximately every two seconds. Utilization measures device
activity, not achieved FLOPs or complete occupancy. Plenty of memory remained;
the test was not memory-capacity limited. Per-episode times generally increased
substantially with four workers. The model server still serializes requests, and
GPU6 shares inference with rendering; resource contention and final-job tail
effects are plausible explanations, not a separately isolated profile result.

| Seed | Result | Actions | 2-env episode seconds | 4-env episode seconds |
|---|---|---:|---:|---:|
| 100000 | success | 328 | 202.847 | 379.398 |
| 100001 | success | 326 | 217.207 | 366.619 |
| 100003 | success | 329 | 165.883 | 392.369 |
| 100004 | failure | 900 | 438.706 | 956.419 |
| 100005 | failure | 900 | 417.107 | 863.018 |
| 100006 | failure | 900 | 420.791 | 857.220 |
| 100008 | failure | 900 | 385.191 | 853.314 |
| 100009 | failure | 900 | 340.961 | 489.127 |

Validation compared final HDF5s **by seed**, not their completion-order filenames:

- All eight seeds completed exactly once; both SQLite queues contain eight done
  rows. Both supervisor ledger/final-observation checks passed.
- Both runs produced three successes, five full 900-step failures and **5,491
  observations**. All **16,473 camera JPEGs** are byte-identical.
- All 11 datasets per episode match exactly: joint/policy actions, three camera
  streams, candidate counts/Qs, selected indices/Qs, margins and transition masks.
- Success, instruction, task/config and final-observation attributes match.
- Ten collection-pool unit tests passed locally and on 190 before both runs.
  Model, loss, inference, sampling and dataset-writer code was not changed.

Authoritative artifacts on 190:

```text
/data3/hongjia/robonana/outputs/collection_pool_dynamic2_8_20260908
/data3/hongjia/robonana/outputs/collection_pool_dynamic4_8_20260908
```

Each contains `summary.json`, `config.json`, `gpu_usage.jsonl`, the SQLite queue,
worker logs/episode ledgers and the separate dataset. Use these ledgers rather
than the stock RoboTwin `_result.txt`: its outer evaluation count is not the
number dynamically claimed by an individual worker. No probe dataset was merged
into replay, and production collection defaults were not changed.

Linear extrapolation at this exact episode-length mix is **4.98 hours per 100
episodes** with two workers versus **5.08 hours** with four. This is not a measured
100-episode duration: seed discovery is excluded, startup/drain costs do not scale
linearly, success/length mix can change, and eight-reset/long-run stability remains
untested. A sensible next controlled test is a dedicated inference GPU6 with two
environment workers on GPU7 (`--sim-gpus 7 7`). Request batching is another distinct
experiment requiring action/Q numerical and selected-index checks; neither was
enabled or claimed faster by this test.

## Request batch 2, candidate group 32: offline probe (2026-09-08)

Implementation/test commit `9bd4ed8`. Nineteen focused tests passed on 190
(collection queue, metrics wrapper, multi-client batching/stress, context cache).
Eleven dependency-free tests passed locally; the local default Python lacks
PyTorch, so model-related local tests cannot be collected. The existing tail
latency test now times request completion separately from server shutdown.

The real step11000 checkpoint was loaded on otherwise idle B200 GPU6. The probe
used first/last planning-boundary observations from one success and one failure
episode, with fixed probe seeds 12345..12348. Each setting warmed the actual policy
and language cache, then processed the same four observations three times. Timing
includes live image encoding, policy/action/Q computation and identical small CPU
copies for verification; simulator work and model startup are excluded.

| Request batch x candidate group | Median time for 4 observations | Peak CUDA allocated / reserved | Throughput versus 1x16 |
|---|---:|---:|---:|
| 1 x 16 | 18.659 s | 31.39 / 32.92 GiB | 1.000x |
| 1 x 32 | 16.905 s | 34.32 / 37.23 GiB | 1.104x |
| 2 x 32 | 16.638 s | 42.14 / 48.43 GiB | 1.121x |

Thus 64 candidate rows fit in memory. Most measured gain came from eliminating
the two candidate groups; cross-request batching added only about 1.6% throughput
over 1x32 in this small warm-cache probe. Do not infer end-to-end collection speed
from this isolated measurement or confuse CUDA allocator peaks with total device
memory occupied by the simulator/driver.

All outputs were finite and selected indices stayed `[7, 6, 24, 13]`. Against
1x16, all-candidate normalized-action maximum absolute error was `3.99e-6` for
1x32 and `0.00175744` for 2x32. Maximum Q error in return units (scale 1000) was
`0.00035763` and `0.23978949`, respectively. This is **not bitwise equivalence**;
the unchanged frozen encoders also execute with a different batch shape. This
probe did not isolate the source of the numerical differences and does not
guarantee unchanged argmax on other observations or identical closed-loop SR.

Raw config/results:
`/data3/hongjia/robonana/outputs/inference_batch2_candidate32_20260908`.
The benchmark calls the existing `BatchedRoboNanaRobotWinPolicy.inference_batch`
and sampler; no model, precision, reward, action count or Q selection math changed.

### Completed live collection

The four-worker 2x32 run at commit `9bd4ed8` finished in **1445.930 seconds**
(24m06s), 19.918 episodes/hour, versus 1462.043 seconds for four workers with
1x16. It issued 49 double-request and 18 single-request calls (116 requests;
84.48% served in double batches). Three episodes succeeded, but the successful
seed set changed: 100000 became a failure and 100006 became a success. Total
control steps were 5447 versus 5483 in the baseline, so raw wall-time differences
are not a clean compute-only speedup. No useful end-to-end gain is established.
Both queue and supervisor data-integrity checks passed. GPU processes exited.
Artifact: `/data3/hongjia/robonana/outputs/collection_pool_batch2_candidate32_8_20260908`.

### Numerical cause isolated (2026-09-08)

Read-only diagnostic commits `0284a8a` / `1c07bd9` used the same real checkpoint
and original four observations/noise seeds. No production precision, encoder,
model, mask or cache implementation was changed. The diagnostic's temporary
backend switch was restored in `finally` and its process exited.

Runtime: PyTorch 2.8.0+cu128, CUDA 12.8; matrix-multiply TF32 disabled,
float32 matmul precision `highest`, VAE parameters FP32, **cuDNN TF32 enabled**.
FP32 tensor storage alone therefore did not establish strict-FP32 convolution.

| Controlled comparison | Maximum absolute difference |
|---|---:|
| VAE image tokens: individual vs paired encoding, deployed backend | 0.00335121 |
| Same comparison with cuDNN TF32 disabled only for diagnosis | 0.0000340343 |
| Cold official Qwen encoding, individual vs two different-length prompts | 0 (tested pair) |
| Original four observations: full 2x32 versus 1x16, normalized candidate actions | 0.00175744295 |
| Same computation, substituting only individually encoded VAE features | 0.00001725554 |
| Original maximum Q difference, return units | 0.239789486 |
| Q difference after that VAE-only substitution | 0.000596046 |

The original action and Q errors were reproduced exactly. Removing batch-dependent
VAE feature differences reduced the maximum action error about 102x and Q error
about 402x. This causally identifies the VAE encoding path as the dominant source
of the observed gap; disabling convolution TF32 reduced the feature gap about 98x.
The experiment does not claim all remaining FP32 rounding disappears.

Additional controls used identical encoded inputs and noise:

- Repeating singleton A produced identical output and all 20 velocities.
- A in `[A,B]` and `[B,A]`, and with B's context/image/state/noise radically
  replaced, showed the same small A error profile: final action `9.54e-7`,
  Q `0.00035763` return units. No large cross-sample influence was observed.
- Adding eight explicitly masked language tokens of value 123 left final action
  error `1.19e-6` and Q error `0.00041723`; no padding-mask failure was observed.
- The existing independent full asymmetric-forward oracle versus cached execution
  on two observations gave action error `4.77e-7`, Q error `0.00011921`.
- Thirteen prefix-cache/full-forward, language-cache and batching regression
  tests passed on 190 CPU. One expected no-CUDA warning was emitted.

With the changed VAE features, the first velocity already differed (A max
`1.42e-4`) and the final velocity differed `9.88e-4`. Flow integration propagates
the upstream discrepancy; new random noise or a changed Euler schedule did not
cause it. These checks identify this measured regression, not a proof that every
possible input is free of all implementation bugs.

Raw diagnostic JSONs:
`/data3/hongjia/robonana/outputs/batch_numerical_diagnosis_20260908/summary.json`
and `/data3/hongjia/robonana/outputs/batch_numerical_aggregate_20260908/summary.json`.

Recommended next bounded implementation test: preserve the existing single-image
VAE encoding path and batch only FLUX/action/Q. This avoids changing the frozen
VAE's historical single-image feature convention. Do not silently toggle TF32
globally: preprocessing/cache and online encoding conventions must be considered
together. No production fix or new collection run was started by this diagnosis.
