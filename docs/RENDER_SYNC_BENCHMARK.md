# Render-sync probe (190, 2026-09-08)

## Decision

Do not enable deferred substep render sync in production collection based on
this test. Both short and full-success replay preserve observations exactly but
have negligible speed benefit. Actual per-control-step image generation/readback
dominates. This corrects the initial hypothesis that redundant sync was a large
collection bottleneck.

## Protocol

- Simulator: existing 190 RoboTwin/SAPIEN/OIDN environment, GPU 7 only.
- Task: `hanging_mug`, `demo_clean`, seed 100000, three RGB cameras.
- Source: `hanging_mug_collection_speed4_20260908`, episode0, recorded actions
  selected by the 11k critic with M=32. Replay does not invoke a policy server.
- Reuse upstream `eval_policy.main` for task/robot/camera configuration and
  upstream `setup_demo`, `take_action`, `get_obs` for simulation.
- Same GPU, fresh sequential processes: baseline, baseline repeat, deferred.
- Keep all control-step observations. Compare uncompressed RGB, joint states,
  success flags. Time action and observation separately; exclude array dumping,
  comparisons, setup and policy inference. Cold initial observation is separate.
- Adapter guards reject dynamic lighting, viewer and video. Nested early-success
  `get_obs` still executes the real synchronization.
- No changes to production client, data frame rate, precision or rendering quality.

## First 96 actions

| Variant | Action seconds | Observation seconds | Sum seconds | Real sync calls |
|---|---:|---:|---:|---:|
| Baseline | 3.4956 | 24.9120 | 28.4076 | 6868 |
| Baseline repeat | 3.5058 | 24.9170 | 28.4227 | 6868 |
| Deferred | 3.2853 | 25.1045 | 28.3898 | 97 |

Deferred skips 6,771 sync calls but saves only 0.06–0.12% in measured replay
time, not a meaningful demonstrated gain. All 97 observations in both comparison
runs match baseline RGB byte-for-byte and states exactly. Both baselines and
deferred also match the original recorded joint states exactly. This truncated
probe does not reach success and alone cannot validate terminal handling.

## Full successful trajectory (328 actions)

| Variant | Action seconds | Observation seconds | Sum seconds | Real sync calls |
|---|---:|---:|---:|---:|
| Baseline | 13.5616 | 85.9742 | 99.5358 | 23829 |
| Deferred | 12.9356 | 85.5688 | 98.5044 | 329 |

Both succeed at action 328. All 329 RGB observations (three cameras), including
the internally captured early-success terminal observation, match byte-for-byte.
Joint states and success flags match exactly between variants. Both differ from
the original stored float32 joint states by at most 2.13e-8.

Skipping 23,500 sync calls saves 1.0314 seconds (1.04%) in this single full replay.
It does not establish a meaningful end-to-end collection speedup: policy inference,
expert checks and setup are excluded and remain unchanged. Short replay only
saved 0.06–0.12%; no repeated full-run confidence interval is available. No failed
full episode or randomized-light scene was validated, and dynamic lighting is
explicitly rejected. Keep the adapter test-only.

## Artifacts and reproduction

Remote output root:
`/data3/hongjia/robonana/outputs/render_sync_probe_20260908`.

Each run stores `result.json`, `rgb.npy`, `states.npy`, `success.npy`, and
`timings.npy`. Generated artifacts are not tracked by Git.

Use `scripts/benchmark_robotwin_render_sync.py --help`. Set
`CUDA_VISIBLE_DEVICES=7`, `OIDN_DEFAULT_DEVICE=cuda`,
`ROBONANA_SAPIEN_RENDER_DEVICE=cuda:0`, and
`ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera`. Supply `--robotwin`, `--episode`,
and a new `--output`; pass `--reference` for comparisons and `--defer` only for
the optimized variant. Omit `--max-steps` for a full trajectory.

Unit tests: `python -m pytest -q tests/test_render_sync_probe.py` (6 passed on
Windows and 190). Test instrumentation is not imported by production collection.
