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
network; only Value is Polyak-averaged, stored/updated in FP32 with BF16
autocast target forward on the shared BF16 FLUX cache.

## Checkpoint boundary

Runtime loading accepts complete `mac_mot_v2` checkpoints. The current 50-task
experiment starts from its own pretraining run; see `MULTITASK_MBRL_PROTOCOL.md`
for exact paths. The older single-task config's default is not this experiment's
initialization. A new critic phase copies the online Value expert into its EMA;
resuming the same critic phase restores the saved EMA.

`scripts/data/convert_120k_action_checkpoint.py` is an explicit one-time export
for the archived actor. The existing export is action-only; its newly initialized
world/critic heads are not trained. The legacy forward, `SegmentMap` and
`build_attention_bias` have been removed; MAC reuses the base modules/helpers.
Conversion is checked against CPU outputs recorded before that removal.
