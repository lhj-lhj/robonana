# World prefix cache

## Execution and lifetime

The maintained `mac_mot_v2` graph permits the following exact decomposition
(in real arithmetic; floating-point implementations are tested separately):

```text
C = [language, state, current image]     prefill once per observation
  -> action denoising / Q candidate scoring
  -> selected G + reward R + success U   prefill once at timestep zero
       -> future state S' + image I'     20 Euler velocity evaluations
```

The clean G/R/U tokens never read S'/I' and have no world-sigma dependence.
Their hidden states, per-layer RoPE-applied K and V, and final binary logits
therefore do not change during denoising. These are predicted tokens, not
ground-truth reward/success labels. We also eliminate the old 21st full
forward that existed only to extract final R/U logits.

`MacFlux2FACTModel._world_suffix` reuses official FLUX block methods, following
the same prepare-QKV / RoPE / mixed attention / residual ordering as the
[pinned ImageWAM MoT adapter](https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/mot.py#L612-L745).
It is not a replacement transformer implementation. Rectangular masks are
sliced from the existing MAC attention graph: R cannot read U, and S' cannot
read I'. Each query uses one joint softmax over its permitted cached and
dynamic keys. Layer caches are detached and request-local, never parameters,
checkpoint fields, or model-global mutable state.

Callers must not reuse a cache after changing observations, clean action,
guidance, precision, or FLUX weights. Stage-1 training continues through the
unchanged full forward with autograd. `sample_mac_world(use_cache=False)` is
kept solely as an independent full-forward oracle; the default is cached.
No denoising steps, targets, terminal threshold, random draws or training
losses are removed or approximated.

## Sharing with critics and precision

`sample_q_rejection(return_condition_cache=True)` exposes its C cache only
when the caller explicitly needs it. Ordinary evaluation results do not
retain a large GPU cache. Stage-2 imaginary rollout passes this same C into
world prefill and returns it for optional reuse inside the wrapped Q/V
forward. Expert gradients still pass through DDP/DeepSpeed's model forward;
only FLUX work is no-grad.

The cache records compute dtype. The critic forward reuses it only when
device and compute dtype match; otherwise it recomputes C. In the current
FP32-regression / BF16-autocast-sampling configuration, regression intentionally
still performs a separate FP32 C prefill. Casting a BF16 cache to FP32 is not
equivalent to computing the prefix in FP32. Thus this change does **not** claim
to eliminate that particular cross-precision computation. World generation
does reuse the sampling C because its execution precision is the same.

Specifically, the saved `hanging_mug_critic_only_5000_to_10000_20260907`
experiment has `train.mixed_precision="no"`. The trainer nevertheless explicitly
wraps **the whole no-grad imagination function** in BF16 autocast, using the
historically named `ema_forward_autocast_dtype`; this includes action/Q
selection, world generation, and both next-state Values, not just EMA Value.
The later differentiable Q/V forward is outside that context. FP32 weight/EMA
storage therefore does not mean every operation runs in FP32. This predates
the cache change. The shared base config separately defaults to `"bf16"`, so
always inspect a run's saved config rather than inferring its precision from
the current base default. Batch-size changes do not change either policy.

Existing BF16 reduced-precision reduction safety settings remain unchanged.
No EMA FLUX, new weights, optimizer state or checkpoint migration is introduced.

## Validation on 190, 2026-09-07

Validated in `/data3/hongjia/robonana/_tmp/world_cache_validation_20260907`,
without replacing the running critic-only experiment's source or process.

- Full CPU regression: **126 passed, 1 skipped**. Includes full-vs-cached
  20-step trajectories in FP32/BF16, mixed/padded language batches, cascade
  masking, detached caches, precision mismatch rejection/recomputation,
  identical Q/V gradients, and two-rank DDP with external cache reuse.
- Real checkpoint: completed stage-2 step 5000 from
  `hanging_mug_mac_critic_from_step10000_5000_20260907`.
- Real data: saved stage-1 config from
  `hanging_mug_mac_world_5000_to_15000_merged100_plus_randomized550_20260906`;
  original success, collected success, and collected failure tail windows.
- B200 GPU 6, FP32 weights with BF16 autocast, 20 steps, B=1, 3 timed repeats:
  6 windows had **zero observed max absolute error** in endpoint image latent,
  future state, reward logits and success logit. Terminal decisions matched.
  Full world rollout took **1.11–1.18 s**, cached **0.585–0.670 s**:
  **1.69–2.02x** faster. This is world-rollout latency, not total Stage-2 step
  speedup. Timings were collected while another training job shared GPU 6;
  they are not an isolated-GPU benchmark or a guarantee for other inputs.
- B=4 (training compute shape, repeated real windows with independent noise),
  3 windows / 2 repeats: again zero observed error for all four outputs.
  Full **3.36–3.62 s** versus cached **1.31–1.44 s**, **2.34–2.76x** speedup.
  Peak allocated memory in this process was about **23.45 vs 23.19 GiB**;
  this optimization mainly saves compute, not model parameter memory.
- Pure FP32 B=1: 3 real success/failure windows also had zero observed error
  in all four outputs. One timed repeat per window gave full **6.05–6.19 s**
  versus cached **1.85–1.98 s**. The small repeat count and shared GPU make
  these secondary timing checks, not a stable throughput estimate.

The benchmark includes G/R/U prefill in cached timing. C prefill is excluded
because Stage-2 action selection has already produced it. Both modes start
from the same input and noise; order alternates between repeats. Run
`scripts/benchmark_mac_world_cache.py --help` for checkpoint/data arguments,
`--precision fp32` for the stricter precision check, and `--batch-size 4` to
exercise the training batch shape (repeated windows, independent noise).

An optimization equivalence check is not a new success-rate evaluation and
does not establish that the world model itself is accurate. Ongoing runs keep
their loaded implementation; newly launched processes use the updated code
after deployment. Do not restart an existing experiment just for this change
without user approval.
