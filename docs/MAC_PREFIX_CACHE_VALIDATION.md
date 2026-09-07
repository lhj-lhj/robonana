# MAC shared-prefix validation

> Historical measurements, not the current precision configuration. RoboNana
> FLUX/Q/V are now FP32-only; frozen external encoders remain unchanged.
> See [current policy](../README.md#numerical-policy).

Validated on 190 on 2026-09-06. Implementation scope: asymmetric critic mask,
request-local C prefill, action sampling across cached Euler steps, Q-only
candidate scoring, joint online V/Q cache reuse, and next-state online/target
Value cache reuse. Default candidate group size is 8; override with
`ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE` or the sampler's explicit argument.

## Correctness checks

The final complete suite on `/data3/hongjia/robonana` passed:

```text
162 passed, 1 skipped in 22.19s
```

`tests/test_mac_prefix_cache.py` covers two double and two single layers, slim
experts, two observations, partially/fully padded language, M=1 and M=5, and
partial candidate groups. It checks:

- Per-layer C K/V equal the C slice of the full asymmetric C/G forward.
- Cached Q values and expert gradients agree with that full reference in FP32.
- Changing one candidate affects neither other candidates nor other observations.
- Cached Euler integration agrees with the full actor forward for the same noise.
- Rejection calls C prefill once and never calls the Value expert.
- Joint critic forward calls C prefill once; expert gradients exist and FLUX
  gradients are absent.
- Two Gloo/DDP ranks complete two optimizer steps with synchronized expert
  gradients using the production joint critic forward.

The existing dynamic-batch tail test initially exceeded its one-second timing
threshold (1.045s). It passed alone and in the final full suite; its code was
not changed.

The real 120k Klein-4B migration and BF16 GPU backward smoke also passed on
GPU 7: 153 compatible tensors loaded, 9 legacy tensors skipped, expert width
1024, both experts have gradients, zero trainable FLUX parameters, no target Q,
and no EMA FLUX. The FP32 EMA Value forward also completed.

## B200 performance measurement

### Current precision policy (2026-09-06 follow-up)

Every FLUX model constructor now sets
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`.
BF16 model storage/autocast remain enabled; only reduced-precision GEMM
intermediate reductions are disabled. Both legacy and MAC loaders inherit
this constructor, covering training ranks and independent inference servers.

With that production setting, the real 120k/B200 synthetic benchmark was
repeated (GPU 5, B=1, 512 language / 288 image tokens, 10 Euler steps,
group=8, one warmup, two repetitions):

| M | Full seconds | Cached seconds | Cached peak GiB | Max action difference | Max Q difference |
|---|---:|---:|---:|---:|---:|
| 1 | 0.244 | 0.181 | 10.39 | 0 | 0 |
| 8 | 1.076 | 0.263 | 11.04 | 0 | 0 |
| 32 | 4.120 | 0.795 | 11.04 | 0 | 0 |

The full suite passed **163 tests, 1 skipped** on 190 after this change.
Exact agreement is evidence for these inputs, not a guarantee for every
GPU/kernel/input shape. The earlier measurements below used PyTorch's default
reduced-precision setting and are retained as historical diagnostic evidence.
Running the reproduction command at the end now uses the current policy.

### Historical measurement (reduced-precision reductions enabled)

GPU 6, NVIDIA B200; actual step-120000 FLUX weights and freshly initialized
Value/Q experts; BF16, B=1, 512 language tokens, 288 image tokens, 48 actions,
10 Euler steps. Inputs are synthetic. Each entry has one warmup and two timed
repetitions; time is the median with CUDA synchronization. Memory is peak
allocated memory including model weights, not peak driver-reserved memory.

The uncached reference repeats C for every candidate and every denoising step,
and evaluates unused Value before Q. It uses the **corrected asymmetric Q
mask**, so this comparison isolates execution cost rather than comparing two
different Q definitions. Timings exclude model loading, camera encoding,
Qwen/AE inference, communication, and environment stepping.

| M | Full reference seconds | Cached group=8 seconds | Speedup | Full peak GiB | Cached peak GiB |
|---|---:|---:|---:|---:|---:|
| 1 | 0.244 | 0.203 | 1.20x | 10.45 | 10.39 |
| 8 | 1.155 | 0.221 | 5.23x | 18.26 | 11.03 |
| 32 | 4.114 | 0.804 | 5.12x | 45.02 | 11.03 |

Group=4 used 10.68 GiB at M=8/32, but took 0.379s/1.403s. Group=8 was selected
as the default for its throughput with only about 0.35 GiB additional peak
allocation on this B=1 measurement. For larger observation batches or smaller
GPUs, group=4 remains available.

BF16 execution is not bitwise invariant to batching and kernel shape changes:

| M | Max absolute action difference, group=8 | Max absolute normalized-Q difference | Cached/reference selected index |
|---|---:|---:|---|
| 1 | 0 | 0 | 0 / 0 |
| 8 | 0.09375 | 0.029296875 | 6 / 6 |
| 32 | 0.171875 | 0.0625 | 5 / 5 |

The reported Q differences include the effect of differences in the denoised
actions. FP32 full-versus-cached comparisons are covered separately by the
tight-tolerance unit tests. Equal selected indices in these measurements do
not guarantee identical BF16 rankings for near-tied candidates elsewhere.
These results establish computation/memory behavior, not hanging-mug success
rate or RL improvement. The changed Q mask still requires trained-Q evaluation.

## Reproduction

From the canonical checkout with the existing dependencies:

```bash
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=src:third_party/FACT:third_party/flux2/src \
/data3/hongjia/conda/envs/robonana/bin/python \
  scripts/benchmark_mac_prefix_cache.py \
  --checkpoint <step-120000 diffusion_pytorch_model.bin> \
  --model-config <matching config.json> \
  --counts 1 8 32 --group-sizes 4 8 --sampling-steps 10 --repeats 2
```
