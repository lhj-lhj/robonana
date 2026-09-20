# 代码与训练参考

当前实验见[实验文档](MULTITASK_MBRL_PROTOCOL.md)。所有新启动都使用[显式JSON配置](../scripts/README.md)，不再使用历史单任务配置/环境变量。以下算法说明与启动参数分离。

## Current architecture

The trainable model is `MacFlux2FACTModel`:

* Phase 1 (`world_policy`): FLUX actor/world parameters train. Successful windows train action BC; successful and failed windows train the world targets.
* Phase 2 (`critic`): the complete FLUX backbone is frozen and only `value_expert` and `q_expert` train. Value has a float32 EMA target; Q has no EMA.
* Both experts read the frozen FLUX per-layer K/V through the ImageWAM-style MoT adapter. Scalar queries and scalar heads are newly initialized; FLUX blocks/modulation are copied or scaled.
* Live inference defaults to Q rejection sampling (`M=32`). The L/S/I prefix is computed once and reused for every candidate. `action_only` disables Q selection for the policy-vs-Q evaluation ablation, using the same checkpoint.

The maintained sequence is:

```text
[language | state | current_image | pred_action | clean_action_chunk |
 reward[48] | success | future_state | future_image]
```

The information boundaries are strict: success sees language/state/current-image/clean-action/reward; future state additionally sees success; future image additionally sees future state. Q sees only language/state/current-image/clean-action. Value sees only language/state/current-image.

## MAC losses

For a real or imagined chunk of 48 actions, `r_t` is the per-step binary reward logit (class 0 decodes to `-1`, class 1 to `0`). The Value target is the fixed-horizon n-step target used by MAC:

\[
L^V(\phi)=\mathbb E\left[\left(V_\phi(\hat s_{t+kn})-\sum_{i=k}^{H-1}\gamma^{(i-k)n}\hat r_{t+in}-\gamma^{(H-k)n}V_{\bar\phi}(\hat s_{t+Hn})\right)^2\right].
\]

With one imagined 48-step chunk (`n=48`, `H=1`) this reduces to the terminal chunk return plus the EMA Value bootstrap. The Q regression is:

\[
L^Q(\phi)=\mathbb E\left[(Q_\phi(s_t,\hat a_t)-\hat r_t-\gamma^{48}V_{\bar\phi}(\hat s_{t+48}))^2\right].
\]

Q is trained from learned-world-model imagined rollouts, not a fabricated real-replay Q label. Value and Q are deterministic functions at evaluation time.

## Training loop

### Checkpoint inference contract

New checkpoints save `transformer/inference_contract.json` beside the exported
weights. It binds their SHA-256 to sampling steps, flow shift, fixed/executed
horizon, reward/discount/return scale, the environment candidate budget, the
separate imagination candidate budget, canonical A statistics content, and the
full image pipeline/VAE/runtime fingerprint. Train candidates (default 8) and
environment candidates (default 32) remain intentionally distinct.

Both phase transitions and same-phase resume validate the contract. Every
online server reads omitted settings from it and rejects conflicting explicit
CLI arguments before model loading. The eval launcher no longer supplies an
independent 20-step default. `action_only` remains an explicit unranked-policy
ablation; batching/chunking performance knobs do not change the total budget.

Missing contracts, different VAE/runtime/statistics, and mismatched weight
fingerprints fail closed. A cache certificate or historical run config does
**not** certify an old checkpoint. Existing checkpoints are not modified.
The public training configuration requires certified source checkpoints. Historical
uncertified adaptation is not silently enabled by environment variables.

Original data selections are explicit `task_globs`; the supplied multi-task example
uses Clean and Randomized. Replay is a separate HDF5 source; all pools retain A.

1. Select an explicit certified checkpoint and prepare replay when entering a new phase.
2. Phase 1 trains the action/world model. BC loss is multiplied by `action_loss_mask`, which is one only for successful trajectories. World losses always run on both success and failure data.
3. For successful terminal windows, observations after the terminal frame are padded as an absorbing state for the remaining chunk suffix. Failure windows are never padded: their final complete chunk ends exactly at the last recorded observation.
4. Phase 2 freezes FLUX, creates/loads the online Value and Q experts from the current Phase-1 checkpoint, and initializes the Value EMA by copying the **online Value expert** (never the previous EMA). The EMA is updated after each optimizer step. Q has no target copy.
5. For each critic batch, the world model performs one on-policy imaginary rollout. Candidate actions are sampled from the action flow, scored by Q, and the argmax candidate is selected. Successful selected environment trajectories are appended to the replay pool and later provide BC updates in the next Phase-1 round.

The default pilot configuration uses one imagined chunk, eight critic candidates during training, 32 candidates in live environment inference, and 20 flow steps. These are configuration knobs, not alternate algorithms.

## Data contract

The canonical dataset is `RoboTwinHDF5Dataset` (the LeRobot adapter reuses the same fixed-48 contract). Each sample contains:

`context`, `current_latents`, `future_latents`, `state`, `future_state`, `behavior_action`, `action_valid_mask`, `reward_chunk`, `reward_chunk_mask`, `success`, `reward`, `reward_h`, and `chunk_horizon=48`.

Default `fixed48` has no sampled horizon. The optional `rope_prefix` world ablation adds scalar sample metadata `world_horizon=h` (uniform 1..48), never a horizon token: image/state/success target t+h, reward independently covers the full chunk, including successful absorbing padding. R uses time coordinate 0 and sees all G; U/S'/I' cannot read R and see only the first h causal G tokens. Both modes retain `chunk_horizon=48` and the same action BC windows. For N observation rows, success starts are `0..N-1`, including the terminal observation, and every step of each 48-action chunk is supervised. Terminal/padded actions hold the final observed pose. Failure starts remain `0..N-49`, with no padding and no action BC; shorter failure episodes are excluded.

Terminal joint targets are `q_final - q_current`, not unconditionally zero: at the terminal observation they are zero in raw delta coordinates and `-action_mean/action_std` after normalization. Grippers keep the final absolute values. The source `transition_valid=False` on the terminal row still means no real transition, but must not suppress its absorbing-state BC target. Collected HDF5's repeated last command is a storage placeholder; terminal BC uses the final observed state. This leaves original successful LeRobot data equivalent to FACT's repeat-last action targets.

## Checkpoints and commands

There is no default trained checkpoint or historical experiment directory.
Use the complete JSON files described in [the configuration guide](../scripts/README.md).

```bash
python scripts/run_multitask_mbrl.py train --config configs/train.json --execute
python scripts/run_multitask_mbrl.py resume --config configs/resume.json --execute
```

Validate a complete checkpoint:

```bash
python scripts/diagnostics/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <checkpoint>/config.json --smoke-forward
```

The loader is intentionally strict: architecture must be `mac_mot_v2`, chunk/reward dimensions must be 48, and all non-expert FLUX parameters must be present. Frozen FLUX parameters are included in the checkpoint-resume protocol so Phase 2 can resume safely under DeepSpeed.

## Repository map

* `src/robonana/models/mac_flux2_fact.py` — fixed-48 FLUX actor/world model.
* `src/robonana/models/flux2_scalar_expert.py` — ImageWAM-style deterministic MoT scalar expert.
* `src/robonana/models/pretrained.py` — strict MAC checkpoint loading and trainable-surface control.
* `src/robonana/data/robotwin_hdf5.py` — success/failure windowing and absorbing-terminal targets.
* `src/robonana/sampling.py` — action flow, one-chunk world rollout, and Q rejection sampling.
* `src/robonana/training/robotwin_trainer.py` — two-phase training and Value EMA updates.
* `src/robonana/configs/training.py` — explicit options, validation and one FACT dictionary assembly.

## Configuration reference

`configs/train.json` is the visible experiment input. All fields are required;
unknown keys/types and inconsistent batch settings fail before launch. No modules
read experiment environment variables at import time. `configs/training.py`
assembles one final FACT config, including synchronized dataset/model horizon,
training/inference sampling, and max_steps/scheduler/checkpoint endpoints.

Model constants remain MAC: action/state14, chunk48, reward48 binary logits,
scalar Q/Value, BF16 FLUX and FP32 VAE/Value EMA storage. These are architecture
contracts rather than independently overridable experiment fields.

Saved-phase continuation is separate from fresh initialization: it reads the
explicit source JSON and prints all inherited settings and requested changes.
See the configuration guide for the batch equation, phase budgets, and files
saved alongside each run.

## Exact phase behavior

### Phase 1 — world/policy

Each minibatch mixes original successful demonstrations, selected-policy
successes, historical failures, and latest failures. An empty optional pool is
redistributed by the sampler instead of failing a round.

For clean action `G` and noisy action `A`, FLUX predicts the action flow target
and, from pure noise, the future image and future state. The 48 reward logits
use this target convention:

```text
real transition before terminal:  class 0 -> reward -1
absorbing terminal suffix:        class 1 -> reward  0
failed time-limit suffix:         invalid mask (not padded)
```

`action_loss_mask=1` only for successful episodes. `reward_chunk_mask` masks
unknown failed tails. Failures therefore train the world model without being
treated as demonstrations; there is no full-model EMA.

### Phase 2 — deterministic critics

The trainer exact-loads the preceding Phase-1 online checkpoint, freezes every
FLUX/world parameter, and enables gradients only for `value_expert.*` and
`q_expert.*`.

At the beginning of a new critic phase:

```text
online Value  <- exact Value expert from loaded checkpoint
online Q      <- exact Q expert from loaded checkpoint
target Value  <- deepcopy(online Value), converted to FP32 for EMA accumulation
```

When resuming the same critic phase, the saved target Value and EMA metadata
are restored. A new phase never initializes from the previous phase's target
Value. After every finite optimizer step, target Value is Polyak-averaged; Q
is never EMA-updated.

Each critic batch performs exactly one learned 48-step imaginary transition.
Candidates are generated by the same action flow sampler used online, Q picks
the best candidate, and that selected action is fed to the world model. With
`R_chunk` decoded from the 48 binary reward logits:

```text
terminal = (sigmoid(success_logit) >= 0.5)
V_target = R_chunk + gamma**48 * (1-terminal) * target_V(next_state)
Q_target = R_chunk + gamma**48 * (1-terminal) * online_V(next_state)
```

Both targets are stop-gradient. Any non-finite loss or gradient flag is
reduced across ranks with a true minimum/AND check; all ranks skip the update
before backward, optimizer, scheduler, or EMA work.

## Q rejection and prefix reuse

World rollouts now reuse `[L,S,I,G,R,U]` per-layer K/V and denoise only
`[future state, future image]` for the original 20 Euler steps. Stage 2 shares
the selected observation's C cache with world prefill. Q/V regression reuses
C only at identical compute precision; the trainer now aligns both passes to
FLUX. Cross-precision external callers recompute rather than cast a cache.
No stage-1 loss, critic target, or
checkpoint format changes. See [world-cache design and validation](WORLD_PREFIX_CACHE.md).

For one request, the frozen FLUX prefix is computed once and shared:

```text
C = FLUX([language, state, current image])
for candidate m in groups:
    A_m = action-flow(C, noise_m)
    Q_m = Q-expert(C, clean_action=A_m)
return action[argmax_m Q_m]
```

L/S/I can attend only to L/S/I; each candidate's clean-action tokens can
attend to C and itself; candidates cannot read one another. Candidate groups
bound peak memory while preserving the full M-way result. The benchmark
script compares cached and uncached M=1/8/32 execution.

RoboTwin evaluation uses `run_multitask_mbrl.py eval --config configs/eval.json`.
`inference_mode` explicitly chooses direct action or deterministic Q rejection.
`capture_mode` chooses SR-only, verified-failure replay, or all trajectories.
All modes share the same worker and preserve checkpoint sampling contracts.

## Replay collection and selected-policy BC

The same evaluation pipeline writes validated replay artifacts. For full capture,
`export_dataset` publishes a flat dataset view; `prepare_robotwin_rollouts.py`
builds the existing index and caches. No second collection launcher is maintained.

## Validation checklist on 190

Before a formal run, record commit, saved config, checkpoint, dataset index,
process command, logs, and GPU ownership. Then run:

```bash
cd /data3/hongjia/robonana
git status --short
nvidia-smi
python -m pytest -q
python scripts/diagnostics/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <checkpoint>/transformer/diffusion_pytorch_model.bin \
  --model-config <checkpoint>/config.json --device cuda:0 --smoke-forward
```

For a bounded world-only probe, explicitly set `smoke_steps` in the training
JSON and inspect the printed resolved budget before execution. Existing pilot
outputs remain untouched; their launchers are not maintained separately.

### W&B credentials on 190

Online W&B authentication is kept outside the repository on the validation
server. The credential is stored in `/home/hongjia/.netrc` with mode `600` and
is read automatically by the server's W&B installation. Never put the API key
in this repository, a config snapshot, a command line, or Git history. Verify
the login on 190 with `wandb login --verify --cloud` without printing the key.

Recommended round order:

```text
1. load the 1,000-step (or previous exact MAC) checkpoint
2. collect selected-policy success/failure rollouts
3. prepare/cache replay data
4. run world_policy
5. inspect fixed-window world fit and finite-loss logs
6. run critic with frozen FLUX and Value-only EMA
7. evaluate M=1 versus M=32 in RoboTwin
```

## Checkpoint and resume protocol

DeepSpeed exports complete FLUX weights even when FLUX is frozen in critic.
Expert and EMA files are stored alongside the transformer export. A checkpoint
is accepted only when architecture metadata, dimensions, reward-head type, and
saved model config agree with `mac_mot_v2`.

If resume reports missing frozen FLUX keys, it is an incomplete pre-fix
checkpoint and must not be force-loaded. Start from the nearest complete MAC
checkpoint. If only target Value is missing at a new critic phase, copy the
loaded online Value expert into a fresh FP32 target; do not copy an unrelated
target from another run.

## Numerical policy

中文：训练和推理统一使用 FACT 官方默认的 BF16；删去 FP32-only 限制及
手工禁用 autocast 的包装。loss、折扣回报、归一化的 FP32 运算仍保留。

Training reuses [FACT's BF16 configuration](https://github.com/Bariona/FACT/blob/9427ea451e806220742148049ef0576e43ef7382/world_action_model/configs/robotwin.py)
and Trainer/Accelerate lifecycle. The loader receives `self.dtype`, and the
three inference services use BF16, matching [FACT's default server](https://github.com/Bariona/FACT/blob/9427ea451e806220742148049ef0576e43ef7382/scripts/inference_server.py).
FACT has no MAC Q/V implementation; our existing experts inherit the same dtype.

| Component | Stage 1 | Stage 2 | Environment inference |
|---|---|---|---|
| world-policy / FLUX | BF16, trainable | BF16, frozen/eval | BF16, eval |
| online Q / V | BF16, frozen | BF16, trainable | Q BF16 for argmax; V normally unused |
| Value EMA | absent | FP32 storage/update, BF16 autocast forward, frozen/eval | unused; same mixed-precision forward if evaluated |

中文：Value EMA 只保留一份 FP32 权重，更新和保存均为 FP32；通过 BF16
autocast 读取已有 BF16 FLUX cache 做混合精度前向，不把 EMA 本体转成 BF16。

Value EMA has one persistent FP32 copy, updated with FP32 `lerp_` and saved
as FP32. Its target forward uses BF16 autocast; it neither recomputes FLUX in
FP32 nor casts cached K/V. This is mixed-precision forward, not pure BF16.
New phases copy online Value into FP32; same-phase resume restores the target.
Legacy BF16 target checkpoints are upcast on load; already-lost increments
cannot be recovered. Online model checkpoint loading still uses BF16.
VAE decoding preserves BN buffer precision (FP32) for sqrt(var+eps) and inverse
normalization before unpatchifying and decoding, matching official FLUX.
Loss/return reductions stay FP32, as do normalization and reporting boundaries.
Stage-1 noise follows FACT's ordering: keep clean action/future state/latents,
sample noise, broadcast sigma and form noisy inputs and velocity targets in
FP32; only the inputs passed to the model are cast to BF16. Targets remain FP32
through loss evaluation. Stage-2 inference noise is generated directly in BF16.
FACT/DeepSpeed owns optimizer/master-state precision. FACT's TF32 default is
retained; model construction no longer changes global BF16 reduction flags.
Inference entrypoints have no separate precision selector.

Validation on 190 for code commit `02b4b4e` (2026-09-09): full CPU regression
191 passed / 2 CUDA-only skips; targeted CPU/B200 training, inference-cache,
EMA, loader and config checks 36 passed, including both skipped CUDA cases.
The two-rank BF16 DeepSpeed nonfinite-gradient diagnostic also passed (NaN
on rank 1 during accumulation). This verifies small-model execution/lifecycle;
the existing full-size FP32 training job was not restarted or benchmarked here.
Frozen Qwen and its language cache are unchanged. Image encoding now uses the
single FACT/FLUX contract below; historical image caches are not silently reused.
Inference sanitizes nonfinite decoded actions with a
finite fallback; action clipping remains disabled. When diagnosing instability,
first lower critic learning rate or candidate count; do not silently add a
second EMA or target Q.

## Unified image pipeline (2026-09-08)

Original LeRobot video, collected HDF5 RGB and live observations all call
`robonana.image_pipeline.build_robotwin_vae_input` and
`robonana.encoding.encode_flux2_image_tokens`:

1. Decode RGB uint8 (live uint8/255 is rounded back to integer pixels).
2. Reuse FACT `scripts/compute_vae_latents.py::_build_composite`: per-view
   bilinear resize, `align_corners=False`, no antialias, then per-view [-1,1]
   normalization and the original 384x192 camera layout.
3. Frozen eval FP32 VAE, posterior **mode**, native batch **one image** even
   when cache I/O or environment requests are batched; scoped TF32 off,
   deterministic cuDNN, benchmark off and autocast off.
4. FLUX packing/BN normalization and BF16 storage rounding. The existing encoder
   returns the same FP32 representation of these rounded values; the shared
   training/inference input boundary casts it to BF16 for FLUX/Q/V.

Image caches now live only in `flux_cache/latents_v2`. The task contract records
VAE weights/config hashes, FACT helper hash and runtime versions; episode
completion metadata proves the shape/dtype/contract. Training checks every
configured dataset pool against its VAE before use. Missing/old/mismatched
caches fail rather than silently mixing versions. Use the existing LeRobot
and HDF5 preprocessing commands with `--stage images` to rebuild explicitly.
Old `latents` files and checkpoints are retained, not overwritten or relabeled.
Batch-size flags now group I/O; VAE execution remains N=1.

All training pools, online policy and replay preparation now use only
`/workspace/datasets/fact-robotwin-v2/RoboTwin/robonana_norm_stats.json` (A).
`robonana.normalization` is the authoritative loader; B and dataset-local
statistics are rejected even when all pools consistently select them.
Replay preparation writes only an episode index and references A, never
recomputes or overwrites a replay statistics file. Newly generated continuation
configs explicitly correct every pool to A; saved source configs stay untouched.
See [the maintained code map and cleanup scope](CURRENT_CODE_MAP.md).

**Historical caveat:** original LeRobot preprocessing and replay preprocessing
previously differed in resize antialiasing. Live encoding also lacked the
cache BF16 roundtrip and had batch-dependent VAE execution. Old checkpoints
were trained on those historical inputs; changing code cannot retroactively
make their training consistent. Validate/retrain with rebuilt caches before
claiming train/live parity. Historical JPEG/MP4 compression is irreversible.
New collected HDF5 RGB uses lossless PNG (schema 4) so replay retains exactly
the pixels supplied to live inference. Old JPEG data remains readable.

`scripts/diagnostics/verify_image_pipeline.py` tests identical decoded frames through cache
and online batch=1/2 with the real VAE; it writes no dataset caches. Equality
on this test does not promise bit-identical results across GPU/runtime changes
or certify action sampling numerics, which are a separate boundary.

190 validation at `1c83eb0`: 155 tests passed, one skipped; real B200/FP32 VAE
on two decoded HDF5 replay frames and two decoded Clean LeRobot frames gave
bitwise-equal pixels and `[2,288,128]` model inputs for cache/live B1/live B2
(max absolute error 0). This is a bounded regression probe, not a full-dataset
audit or proof of old-cache parity. See [the audit](archive/IMAGE_PIPELINE_AUDIT.md).

## Archived history

The original 120k checkpoint remains outside the maintained runtime path for reproducibility. Its actor export is a one-time conversion; the legacy mask/forward is removed. The new `rope_prefix` ablation uses the current MAC implementation and RoPE, not the old explicit horizon-token architecture.
