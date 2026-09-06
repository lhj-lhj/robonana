# Model boundary and upstream references

RoboNana reuses the official FACT/FLUX.2 implementation and adds only the
fixed-48 RoboTwin wiring. The maintained model is `MacFlux2FACTModel`.

| Source | Reused | RoboNana extension |
|---|---|---|
| FACT | trainer lifecycle, samplers, collators, Accelerate/DeepSpeed | fixed-48 RoboTwin data and losses |
| FLUX.2 | projections, RoPE, modulation, transformer blocks | action/state/world token layout |
| MAC | one imagined rollout, Q action selection, Value-only target | RoboTwin reward/world wiring |
| ImageWAM | cached MoT K/V expert structure and initialization policy | deterministic scalar Value/Q experts |

References: [MAC](https://github.com/kwanyoungpark/MAC), [ImageWAM](https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a).

## Fixed-48 information flow

```text
[L | S | I | A_pred | G_clean | R[48] | success | S' | I']
```

The world cascade is `R -> success -> S' -> I'`. The action track is not
visible to world targets except the clean action conditioning track `G`.
Value reads only `[L,S,I]`; Q reads `[L,S,I,G]`. Both are deterministic,
one-query experts. FLUX is frozen during critic training. Q has no target
network; only Value is Polyak-averaged in float32.

## Checkpoint boundary

Runtime loading accepts only complete `mac_mot_v2` checkpoints. New runs use
the 1,000-step MAC checkpoint configured in `posttrain_config.py`; later rounds
exact-load the previous online model and initialize a fresh Value EMA as a copy
of that online Value expert. Resuming the same critic phase restores its saved
EMA state. The old 120k checkpoint remains an archived artifact, not a runtime
format, and no conversion/legacy loader is shipped here.
