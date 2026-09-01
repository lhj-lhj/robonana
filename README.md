# RoboNana

RoboNana is a RoboTwin 2.0 world-action model built by extending FACT's data and
training path with one shared FLUX.2 DiT. The repository keeps upstream FACT and
FLUX.2 implementations external and contains only the adapters, masks, cache
contracts, losses, training hooks, and RoboTwin integration needed by the current
experiment.

## Current experiment

The canonical pretraining entrypoint is
`robonana.configs.robotwin_flux2_4b_dino.config`:

- official pretrained FLUX.2 Klein 4B backbone;
- hidden size 3072, 24 attention heads, 5 double-stream blocks, and 20
  single-stream blocks;
- BF16 training on eight GPUs with ZeRO-2 and gradient checkpointing disabled;
- 120,000 optimizer steps at global batch size 256 (`16 x 8 x 2` gradient
  accumulation);
- FACT RoboTwin-v2: 2,500 Clean episodes plus 25,000 Randomized episodes;
- language, robot state, three-view current image, action, horizon, future state,
  future-state reward, terminal-success logit, Q, future FLUX-AE image, and a
  training-only future DINOv3 target;
- pretrained-backbone learning rate `2e-5` and RoboNana adapter/head learning
  rate `1e-4`;
- fixed-horizon W&B visualization at `h = 12, 24, 48` every 2,000 steps.

Training samples `idx_h` uniformly from `1..48`. A target beyond an episode uses
the final valid frame/action, which matches the retained FACT tail-clipping
behavior. The scratch 800M configuration is retained only for inexpensive
contract tests, smoke runs, and ablations; it is not the default experiment.

## Shared-DiT design

The exact token order is:

```text
[language | state | current_image | pred_action | gt_action_full_clean |
 idx_h | future_state | reward | success | Q | future_image_vae | future_image_dino]
```

All segments pass through the same FLUX.2 double-stream and single-stream
blocks. Image, action, future-state, reward, success, Q, and DINO predictions use separate
output projections; they are not separate transformer backbones. The DINO
branch adds only two projections whose middle dimension follows the selected
shared DiT hidden size:

```text
4B:   dino_in/out = Linear(3072, 3072) / Linear(3072, 3072)
800M test: dino_in/out = Linear(3072, 1536) / Linear(1536, 3072)
```

The scalar branches are equally small adapters around that same hidden stream:

```text
learned reward query [B,(K),1] -> shared FLUX.2 DiT
                                -> reward_out Linear(hidden, 1, bias=False)

learned success query [B,(K),1] -> shared FLUX.2 DiT
                                 -> success_out Linear(hidden, 1, bias=False)

noisy Q      [B,(K),1] -> q_in Linear(1, hidden, bias=False)
                         -> shared FLUX.2 DiT
                         -> q_out Linear(hidden, 1, bias=False)
```

Reward is trained by scalar regression and success by BCE-with-logits; neither
is flow-noised. Q remains a scalar flow token. They have distinct queries and
output heads, but no independent scalar backbone. `K` is present only when multiple horizon blocks are packed
for parallel Stage-2 inference.

The attention policy is prefix-structured:

| Query | Visible keys |
|---|---|
| language, state, current image | language, state, current image |
| predicted action token `A_t` | clean prefix + full `A_1..A_48` chunk |
| full-clean GT action token `G_t` | clean prefix + `G_1..G_t` |
| horizon | clean prefix + `G_1..G_idx_h` + horizon |
| future state | previous world prefix + future state |
| direct reward | previous world prefix + direct reward |
| success logit | previous world prefix + success logit |
| Q (MC in pretraining, TD in posttraining) | previous world prefix + Q |
| future FLUX latent | previous world prefix + future FLUX latent |
| future DINO | complete world prefix + future DINO |

The noisy predicted-action track A is bidirectional inside its 48-token chunk,
so diffusion denoises the complete action trajectory jointly. The full-clean
conditioning track G is causal: token `t` cannot read token `t+1`. For every
sample, horizon, future state, reward, success, Q, future FLUX latent, and future DINO can
read only the first `idx_h` full-clean action tokens.
The mask is constructed dynamically from the batch's `idx_h` tensor. The
predicted-action track remains a sink and is never visible to the GT/world path.
DINO is a final one-way auxiliary sink, so earlier tokens cannot depend on it.
Inference omits the DINO suffix and uses the unchanged action and future-image
samplers.

### What reward, success, accumulated reward, and Q mean

The two scalar heads have different semantics. Let the sampled current frame be
`t`, let the requested horizon be `h = idx_h`, and define

$$
t_h = \min(t+h, T-1), \qquad
\delta = \sum_{j=t}^{t_h-1}\mathrm{transition\_valid}_j.
$$

For an original successful demonstration every transition is valid, so
`delta = t_h - t`. Replay episodes use the stored `transition_valid` vector so
clipped padding and a reset observation never count as real robot transitions.
`future_state`, future FLUX-AE latent, and future DINO target are all the single
observation at `t_h`; they are not the last element of the fixed 48-action
chunk.

RoboNana uses `gamma = 0.999`. The model's direct `reward_h` output is attached
to the selected future state: `-1` for a nonterminal state and `0` for the true
successful terminal state. `success_h` is a redundant Bernoulli logit for that
same terminal event. This intentional duplication provides an explicit
terminal detector while retaining a scalar reward interface for later tasks
with non-binary rewards.

The dataset separately preserves the accumulated prefix reward (the batch
field remains named `reward_h`) for TD target construction. It is the discounted
real reward over the valid behavior-action prefix that reaches `t_h`:

$$
R_t^{(\delta)} = \sum_{k=0}^{\delta-1}\gamma^{k} r_{t+k}.
$$

With the current constant step cost this is `0` when `delta=0`, otherwise

$$
R_t^{(\delta)} = -\frac{1-\gamma^{\delta}}{1-\gamma}.
$$

For example, a nonterminal `h=48` target still has direct reward `-1` and
success target `0`, while its accumulated TD prefix reward is `-46.889103` when
all 48 transitions are valid. At a clipped successful terminal, direct reward
is `0` and success is `1`; clipped padding adds no transitions to the accumulated
prefix reward.

The meaning of `Q` depends on the training stage, while its token, adapter,
head, flow-matching path, and inference API remain unchanged:

#### Successful-demonstration pretraining

The target is the complete Monte Carlo return from the current frame under the
recorded behavior-policy continuation:

$$
Q_t^{\mathrm{MC}} =
\sum_{k=0}^{T-1-t}\gamma^{k} r_{t+k}.
$$

For a fixed `t`, different sampled horizons can change the direct reward/success
label and the accumulated prefix reward, but the MC-Q label is the same. At the
successful final frame, direct reward and Q are zero and success is one.

#### Iterative posttraining

The target is a bootstrapped action-chunk Q:

$$
y_t^{Q} = R_t^{(\delta)} +
\gamma^{\delta} (1-d_h^{\mathrm{success}})
Q_{\bar{\theta}}(s_{t_h},\pi_{\bar{\theta}}(s_{t_h}); h=48).
$$

The current Q token is conditioned on the current observation and the first
`h` tokens of the full-clean **recorded behavior action** track `G`. Thus it
means: discounted return for executing that real action prefix, followed by
the EMA policy continuation used to build the TD target. A successful
terminal stops the bootstrap; a failure timeout does not.

No scalar is min-max normalized or the old time-to-go target. Q is not a
calibrated success probability; under the negative step cost a larger Q (less
negative and closer to zero) is preferred. Scalar outputs are `[B, 1, 1]` for
one horizon and `[B, K, 1]` for `K` packed horizons. Inference exposes direct
`rewards`, sigmoid `success_probs`, and sampled `qs` after flattening the last
singleton dimension.

See [docs/INHERITANCE.md](docs/INHERITANCE.md) for the exact upstream reuse
boundary.

## Repository layout

```text
src/robonana/models/       shared FLUX.2 wrapper, attention mask, strict checkpoint config
src/robonana/data/         LeRobot-v2 loader, rollout HDF5 loader, cache and stats contracts
src/robonana/training/     FACT trainer hook, joint flow losses, W&B visualization
src/robonana/inference/    two-stage RoboTwin policy; no DINO inference path
src/robonana/sim/          SAPIEN/OIDN runtime compatibility
src/robonana_robotwin*     FACT/RoboTwin TCP adapter and client
scripts/                   current preprocessing, training, evaluation, and rollout entrypoints
tests/                     maintained unit/integration contract tests
```

The supported entrypoint roles are:

| role | entrypoint | DINO behavior |
|---|---|---|
| canonical pretraining | `robonana.configs.robotwin_flux2_4b_dino.config` | frozen DINOv3 targets and DINO flow loss enabled |
| canonical inference | `scripts/inference_server_robotwin.py` with the trained 4B checkpoint | no DINO encoder, target tokens, sampling, or output |
| iterative posttraining | `robonana.configs.robotwin_flux2_4b_dino_posttrain.config` | same training-only DINO supervision |
| test/smoke only | `robonana.configs.robotwin_flux2_800m_dino.config` | smaller scratch backbone for fast validation |

“No-DINO inference” describes the execution graph, not a second loosely loaded
checkpoint format. Strict loading still reconstructs the exact 4B training
architecture, including the small DINO adapter parameters stored in the
checkpoint, but every inference mode omits DINO tokens and never constructs or
runs the frozen DINOv3 encoder. This preserves strict checkpoint diagnostics
while removing DINO computation from inference.

## Environment and upstream code

The canonical checkout on `hongjia@208.64.254.190` is:

```text
~/robonana -> /data3/hongjia/robonana
```

Create an environment and make the upstream repositories importable:

```bash
cd ~/robonana
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[train,preprocess,dev]'

git clone https://github.com/Bariona/FACT.git third_party/FACT
git clone https://github.com/black-forest-labs/flux2.git third_party/flux2

export PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
```

Place the official FLUX.2 Klein component checkpoint at
`checkpoints/FLUX.2-klein-base-4B` or set `ROBONANA_FLUX_CHECKPOINT_DIR`. It must
contain the local Qwen3 text encoder and FLUX AE used for offline caches and
pixel decoding. Download the DINOv3 ViT-B/16 weights into the Hugging Face cache
with the current `hf` CLI:

```bash
hf auth login
HF_HOME=/data3/hongjia/hf-cache \
  hf download timm/vit_base_patch16_dinov3.lvd1689m
```

Never put Hugging Face or W&B tokens in Git. Authenticate W&B once with
`wandb login`; runs use project `robonana` and entity
`hongjia-liu-aalto-university`.

## Dataset and offline caches

The full dataset root on the current server is:

```text
/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2/RoboTwin
```

Per episode, preprocessing writes:

```text
flux_cache/language/episode_NNNNNN.pt  BF16 [512, 7680]
flux_cache/latents/episode_NNNNNN.pt   BF16 [T, 288, 128]
```

The FLUX cache uses the same three-view `384 x 192` composite geometry as the
training input. DINO is deliberately not cached. The loader decodes only the
three native RGB frames at `t_h`; a frozen DINOv3 ViT-B/16 on each training rank
computes `14 x 14 x 768` patch features online and applies lossless `2 x 2`
pixel-unshuffle. This gives `3 cameras x 7 x 7 = 147` tokens with feature width
`4 x 768 = 3072`. The encoder is excluded from the optimizer and RoboNana
checkpoints. This removes the roughly 4.99 TiB full-dataset DINO cache, at the
cost of three random MP4 frame reads and one frozen ViT forward per sample.

Build the metadata and the shared Qwen3/FLUX caches before launching the
canonical 4B run:

```bash
export ROBONANA_DATASET_ROOT=/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2/RoboTwin
export ROBONANA_FLUX_CHECKPOINT_DIR=$PWD/checkpoints/FLUX.2-klein-base-4B

.venv/bin/python scripts/compute_robotwin_lerobot_metadata.py \
  --dataset-root "$ROBONANA_DATASET_ROOT" \
  --task-glob 'Clean/*' --task-glob 'Randomized/*'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "$ROBONANA_DATASET_ROOT" \
  --checkpoint "$ROBONANA_FLUX_CHECKPOINT_DIR" \
  --stage all --batch-size 64 --language-batch-size 4

.venv/bin/python scripts/validate_robotwin_lerobot_flux.py \
  --dataset-root "$ROBONANA_DATASET_ROOT"
```

`scripts/run_full800m_pipeline.sh` remains available only as the historical
800M smoke/ablation automation; it is not the canonical training launcher.

For a bounded cache smoke test, use `--max-tasks` or `--max-episodes` explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "$ROBONANA_DATASET_ROOT" \
  --checkpoint "$ROBONANA_FLUX_CHECKPOINT_DIR" \
  --stage all --max-episodes 1 --batch-size 16 --language-batch-size 1
```

## Training

After the Qwen3 and FLUX cache families validate, the default launcher runs the
pretrained FLUX.2 Klein 4B+DINO configuration. No explicit `--config` is
required:

```bash
export ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7
export ROBONANA_BATCH_SIZE=16
export ROBONANA_MAX_STEPS=120000
export ROBONANA_PIXEL_EVAL_INTERVAL=2000
export ROBONANA_CHECKPOINT_INTERVAL=1000
export ROBONANA_BACKBONE_LR=2e-5
export ROBONANA_ROBOT_LR=1e-4
bash scripts/run_robotwin_train.sh
```

This is equivalent to passing
`--config robonana.configs.robotwin_flux2_4b_dino.config` explicitly. It uses
ZeRO-2, no gradient checkpointing, local micro-batch 16, and two accumulation
steps (`16 x 8 x 2 = 256` global batch). AdamW applies `2e-5` to the pretrained
FLUX backbone and `1e-4` to RoboNana heads, token embeddings, and DINO adapters;
both groups share the same warmup/cosine multiplier.

To warm-start the direct reward/success heads from the existing 150k reward/Q
checkpoint while preserving the trained FLUX, action, state, Q, image, and DINO
weights, run the continuation. On two B200s, local batch 16 gives global batch
32 while leaving safe activation headroom:

```bash
export ROBONANA_GPU_IDS=6,7
export ROBONANA_BATCH_SIZE=16
export ROBONANA_ADDITIONAL_STEPS=10000
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_4b_dino_reward_success_q_from150k.config
```

The old flow-matched `reward_in/reward_out` tensors are explicitly skipped with
a warning; the direct reward query/head and success query/head are newly
initialized. This is an initialization run, not an optimizer-state resume.

The 800M scratch+DINO path must always be requested explicitly and is intended
only for tests, smoke runs, and small ablations:

```bash
export ROBONANA_BATCH_SIZE=32
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_800m_dino.config
```

The loss weights are image `1.0`, action `10.0`, future state `0.4`, direct
reward `0.01`, success BCE `0.01`, Q `0.001`, and DINO `0.1`. Periodic training
visualization performs only Stage 2: it conditions on the batch's full-clean GT
action, samples future image/state/Q from pure noise with 20-step Flow-Euler,
then evaluates direct reward/success once on the clean sampled world. It does
not run Stage-1 action diffusion.

Checkpoint loading is strict. Evaluation must find a complete FACT `config.json`
next to the experiment checkpoint or receive `--model-config`; model size is
never guessed from checkpoint tensor shapes.

## Inference modes

`scripts/inference_server_robotwin.py --inference-mode <mode>` exposes four
strict graphs:

| mode | action source | Stage-2 horizons | future image |
|---|---|---|---|
| `action` | Stage-1 diffusion | none | none |
| `action_reward_q` | Stage-1 diffusion | `h=T` first; expand `1..T` only if `success_T` is positive | omitted |
| `world_all` | request `action_chunk` | all `1..T` in packed horizon batches | FLUX latent + decoded pixels |
| `world_horizon` | request `action_chunk` | request scalar `horizon` | FLUX latent + decoded pixels |

`action_chunk` is the absolute robot-space `[T, action_dim]` chunk; the policy
applies the same delta conversion and z-score normalization used in training.
Packed Stage-2 uses one shared clean causal action track followed by isolated
`[idx_h | future_state_h | reward_h | success_h | Q_h | future_image_h]` blocks. A block sees only
`G_1..G_h` and itself, never another horizon block. State/reward/success/Q-only inference
creates zero future-image tokens and never invokes the VAE decoder. DINO is a
training-only branch and is absent from every inference mode. Image horizons
default to four isolated blocks per forward because packing 48 full FLUX image
grids into one dense-attention sequence is impractical; configure this with
`--stage2-image-horizon-batch-size`.

`action_reward_q` short-circuits the common failure case. It first evaluates
only `h=T`. If `sigmoid(success_T)` is below `--success-threshold`, every earlier
state must also be nonterminal under tail clipping, so the cumulative reward is
computed analytically as `sum_{k=0}^{T-1} gamma^k * (-1)` and no dense horizon
query is run. If `success_T` is positive, horizons `1..T-1` are packed, the
first positive success logit identifies the terminal frame, and the directly
predicted per-frame rewards are discounted through that frame.

Run from the repository root. Define the shared arguments once, then launch
exactly one of the four commands:

```bash
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin
export ROBONANA_FLUX_CHECKPOINT=/path/to/FLUX.2-klein-4B
export ROBONANA_STATS_PATH=/path/to/robonana_norm_stats.json

ROBONANA_SERVER=(
  python scripts/inference_server_robotwin.py
  --checkpoint "$ROBONANA_TRAINED_CHECKPOINT"
  --flux-checkpoint-dir "$ROBONANA_FLUX_CHECKPOINT"
  --stats-path "$ROBONANA_STATS_PATH"
  --model-device cuda:0
  --vae-device cuda:0
  --text-encoder-device cpu
  --dtype bf16
  --action-chunk 48
  --num-inference-steps 20
)

# 1. Stage-1 only: fastest action inference.
"${ROBONANA_SERVER[@]}" --inference-mode action --port 8094

# 2. Stage-1 plus conditional state/reward/success/Q; no future VAE/DINO tokens.
"${ROBONANA_SERVER[@]}" --inference-mode action_reward_q --port 8094

# 3. Supplied action chunk plus all state/reward/success/Q/VAE/pixel horizons.
"${ROBONANA_SERVER[@]}" \
  --inference-mode world_all \
  --stage2-image-horizon-batch-size 4 \
  --vae-decode-batch-size 4 \
  --port 8094

# 4. Supplied action chunk and h plus one state/reward/success/Q/VAE/pixel prediction.
"${ROBONANA_SERVER[@]}" --inference-mode world_horizon --port 8094
```

Requests to `world_all` must include an absolute robot-space
`action_chunk` tensor with shape `[48, action_dim]`. `world_horizon` requires
the same `action_chunk` plus one integer `horizon` in `[1, 48]`. Both modes
still require the normal observation state, three RGB views, and instruction.
If `model_config.json`/`config.json` is not discoverable above the checkpoint,
add `--model-config /path/to/model_config.json` to `ROBONANA_SERVER`.

For closed-loop inspection from training-set first frames, the dataset rollout
script alternates Stage-1 action sampling and `world_all`, decodes all 48 image
horizons, overlays each horizon-specific reward and Q, and feeds the raw decoded
`h=48` three-view pixels and predicted state into the next round:

```bash
python scripts/rollout_dataset_world_all.py \
  --checkpoint /path/to/checkpoint/transformer/diffusion_pytorch_model.bin \
  --model-config /path/to/experiment/config.json \
  --flux-checkpoint-dir /path/to/FLUX.2-klein-base-4B \
  --dataset-root /path/to/fact-robotwin-v2/RoboTwin \
  --output-dir outputs/dataset_world_all_rollout \
  --trajectory-count 10 --rollout-rounds 5 \
  --stage2-image-horizon-batch-size 2
```

## Full RoboTwin evaluation

### Official RoboTwin/XPolicyLab batch path on H100

For Hopper, keep the simulator and scheduler at an unmodified official
RoboTwin commit. RoboNana supplies only a legacy-TCP transport adapter because
the official evaluator and the existing FACT server use different wire
formats. Each official batch worker owns one SAPIEN process and one persistent
TCP connection. Its observation is connection-local; concurrent `get_action`
calls are combined into one `BatchedRoboNanaRobotWinPolicy.inference_batch`
call.

The official observation is converted exactly as follows:

```text
cam_head.color        -> observation.images.cam_high
cam_left_wrist.color  -> observation.images.cam_left_wrist
cam_right_wrist.color -> observation.images.cam_right_wrist

state = [left_arm_joint_state, left_ee_joint_state,
         right_arm_joint_state, right_ee_joint_state]
```

Expose the versioned client shim inside the official XPolicyLab policy tree;
do not copy or patch any RoboTwin source file:

```bash
export ROBOTWIN_ROOT=/workspace/hongjia/RoboTwin-official
mkdir -p "$ROBOTWIN_ROOT/XPolicyLab/policy/RoboNana"
ln -sfn "$PWD/integrations/xpolicylab/policy/RoboNana/deploy.yml" \
  "$ROBOTWIN_ROOT/XPolicyLab/policy/RoboNana/deploy.yml"
ln -sfn "$PWD/integrations/xpolicylab/policy/RoboNana/setup_eval_env_client.sh" \
  "$ROBOTWIN_ROOT/XPolicyLab/policy/RoboNana/setup_eval_env_client.sh"

# Current upstream imports legacy TCP from client_server.model_client although
# its implementation lives at client_server/tcp/model_client.py. This external
# namespace shim re-exports that official class without editing RoboTwin. The
# versioned setup script resolves and passes the shim path automatically.
```

Start the policy on a GPU that is not assigned to SAPIEN:

```bash
python scripts/inference_server_robotwin_xpolicylab.py \
  --checkpoint "$ROBONANA_TRAINED_CHECKPOINT" \
  --model-config /path/to/experiment/config.json \
  --flux-checkpoint-dir "$ROBONANA_FLUX_CHECKPOINT" \
  --stats-path "$ROBONANA_STATS_PATH" \
  --xpolicylab-root "$ROBOTWIN_ROOT/XPolicyLab" \
  --model-device cuda:0 --vae-device cuda:0 \
  --max-batch-size 7 --max-batch-wait-ms 100 --port 8094
```

Then run the official config-driven evaluator on disjoint simulator GPUs. The
official `--eval-batch --num-workers N` mode creates `N` independent SAPIEN
workers per task; it does not create one vectorized SAPIEN scene:

```bash
cd "$ROBOTWIN_ROOT"
export PATH="/path/to/robotwin-env/bin:/path/to/conda/condabin:$PATH"
bash scripts/eval_policy.sh multitask \
  --config /path/to/eval_tasks.yml \
  --policy-name RoboNana \
  --env-cfg-type aloha_agilex \
  --eval-env-conda-env /path/to/robotwin-env \
  --enable-remote \
  --policy-server-ip 127.0.0.1 --policy-server-port 8094 \
  --eval-batch --num-workers 7 --test-num 50 \
  --task-config demo_clean --action-type joint \
  --output-dir /path/to/eval-output
```

`eval_tasks.yml` uses the official scheduler schema, for example:

```yaml
gpu_ids: [1, 2, 3, 4, 5, 6, 7]
jobs_per_gpu: 1
num_workers: 7
tasks: [hanging_mug]
```

One scheduler job receives one `CUDA_VISIBLE_DEVICES` value; every worker for
that task shares the selected simulator GPU. The outer scheduler distributes
different tasks across `gpu_ids`. Tune `num_workers` for the per-GPU SAPIEN
memory budget rather than assuming its workers are spread across the list.

The H100 validation environment on west2 uses the official RoboTwin checkout
at commit `30954692d06ba7e89f7a6b76064f4062c488fa81` and keeps its source
unchanged. The non-interactive launcher needs the conda command, Python,
ffmpeg, and the isolated GLVND runtime on its environment paths:

```bash
export PATH=/workspace/hongjia/.venvs/robotwin-h100-test/bin:\
/workspace/.conda/condabin:/workspace/.conda/envs/dreamdojo/bin:$PATH
export LD_LIBRARY_PATH=/workspace/hongjia/.runtime-libs/robotwin-glvnd/usr/lib/x86_64-linux-gnu\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

A two-worker `click_bell` infrastructure smoke completed two 400-step
episodes with official RT+OIDN and wrote both MP4s. The zero-action smoke policy
produced four true `batch_size=2` model calls; request skew flushed ten tail
calls at `batch_size=1`. This validates the transport and simulator concurrency,
not policy success or full-checkpoint H100 memory.

### FACT-compatible isolated-GPU path

The existing launcher follows the FACT RoboTwin protocol:

- SAPIEN nightly ray tracing with OIDN 2.4.1 CUDA denoising and serialized
  Vulkan/CUDA hand-offs;
- Stage-1 action diffusion only;
- 48 actions executed per policy request;
- only the three policy inputs are rendered: left wrist, right wrist, and head;
- policy servers and SAPIEN/OIDN simulators on disjoint physical GPU pools;
- one fresh SAPIEN process per accepted episode with deterministic seed handoff;
- per-task and aggregate success rates plus native episode MP4s.

The default topology uses GPU 0–3 for four policy servers and GPU 4–7 for four
single-client SAPIEN/OIDN simulators. The launcher rejects overlapping pools;
putting FLUX inference and Vulkan/OIDN on the same physical GPU can trigger an
NVIDIA graphics context-switch timeout and `vk::DeviceLostError` on Blackwell.
Each episode has its own watchdog. A failed CUDA/OIDN attempt is
terminated and retried from the same seed. All retries remain on CUDA; exhausting
them marks an infrastructure error instead of silently changing the renderer.
The denoiser is never disabled, accepted seeds are never silently skipped, and
completed episodes survive task restarts.

```bash
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin
export ROBONANA_DATASET_ROOT=/workspace/datasets/fact-robotwin-v2/RoboTwin
export ROBONANA_EVAL_SERVER_GPUS=0,1,2,3
export ROBONANA_EVAL_SIM_GPUS=4,5,6,7
export ROBONANA_EVAL_JOBS_PER_GPU=1
export ROBONANA_EVAL_BATCH_WAIT_MS=100
export ROBONANA_EPISODE_TIMEOUT_SECONDS=3600
export ROBONANA_EPISODE_GPU_ATTEMPTS=2
export ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

The second positional argument is the number of episodes per task. To rerun a
strict subset without changing the protocol:

```bash
export ROBONANA_EVAL_TASKS=click_bell,stamp_seal
export ROBONANA_TASK_TIMEOUT_SECONDS=43200
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

Each run directory contains `results.csv` with one row per task, `summary.txt`
with micro/macro success rates, `mp4_manifest.txt`, per-worker logs, and the
instruction audit. Each task also has an append-only `episodes.jsonl` seed and
success ledger plus `attempts.jsonl` with timeout/fallback diagnostics. An
`ERROR` row is infrastructure failure or timeout and must be rerun; it is never
counted as a 0% policy result.

On Blackwell, install the pinned SAPIEN nightly plus the serialized GPU OIDN
runtime once in the RoboTwin environment. The build root is persistent because
the patched renderer records runtime paths to its exact dependency build. The
launcher verifies the install manifest, renderer checksum, svulkan commit, and
all CUDA OIDN libraries before starting an evaluation:

```bash
CUDA_PATH=/usr/local/cuda-13.0 \
CMAKE_BIN=/path/to/cmake \
bash scripts/install_sapien_oidn_blackwell.sh \
  /path/to/robotwin2 \
  /persistent/path/robonana-sapien-oidn
export ROBOTWIN_CONDA_ENV=/path/to/robotwin2
```

`SVULKAN_SOURCE_DIR` and `SVULKAN_BUILD_DIR` may point to an existing exact
checkout/build; the installer still verifies the pinned commit, reapplies or
verifies the repository patch, and rebuilds the required targets.

The low-memory renderer test exercises the same Vulkan-to-CUDA OIDN transfer
without loading the RoboNana policy:

```bash
CUDA_VISIBLE_DEVICES=0 OIDN_DEFAULT_DEVICE=cuda \
PYTHONPATH=$PWD/src ROBONANA_SAPIEN_RENDER_DEVICE=cuda:0 \
/path/to/robotwin2/bin/python scripts/stress_sapien_oidn.py --frames 1000
```

## Failure rollout collection

Rollouts are always written outside the initial dataset. The current collector
saves aligned three-view RGB, observed state, actually executed action,
success/failure metadata, Qwen3 context, and FLUX AE cache into a separate HDF5
collection. Every episode also stores the reset-pre final observation, a
`transition_valid` vector, `round_id`, `policy_checkpoint`, and
`policy_version`; a timeout is a truncation and never a reset observation:

```bash
export ROBONANA_INITIAL_DATASET_ROOT="$ROBONANA_DATASET_ROOT"
export ROBONANA_STATS_SOURCE="$ROBONANA_DATASET_ROOT/robonana_norm_stats.json"
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin
export ROBONANA_COLLECTION_ROUND=0
export ROBONANA_POLICY_VERSION=theta_0

TEST_NUM=1 PORT=8095 \
bash scripts/collect_prepare_robotwin_rollouts.sh \
  beat_block_hammer demo_clean step1000_failures 0
```

The metadata/index builder preserves all of those fields. Replay preprocessing
must complete Qwen3/FLUX caches before posttraining; DINO remains online and is
not cached. `ROBONANA_EVAL_RUN_DIR` can route the evaluation ledger/results to
an experiment directory, while `EVAL_VIDEO_LOG=1` makes the same simulator pass
also retain MP4s. The focused posttraining pipeline uses both options so its
baseline success rate, videos, and trainable replay come from exactly the same
episodes.

## Iterative posttraining

Posttraining changes the source of the Q label and adds policy improvement on
failed data; it does not add a second network architecture. The online model
`theta` and target model `theta_bar` are complete copies of the same shared
FLUX.2+RoboNana model. One round has the following data flow:

```text
theta_k checkpoint
  -> collect RoboTwin success/failure rollouts into a separate replay root
  -> write aligned RGB/state/executed-action/final-observation metadata
  -> build Qwen3 context + FLUX-AE caches and robonana_index.json
  -> train theta_(k+1) from four logical pools with EMA theta_bar
  -> save online, optimizer, scheduler, trainer, EMA, and round state
```

The implementation remains on the existing FACT/FLUX paths:

| responsibility | maintained implementation |
|---|---|
| rollout schema and atomic episode publication | `src/robonana/data/rollout_writer.py` |
| horizon clipping, valid-transition count, reward/MC labels, replay views, four-pool sampler | `src/robonana/data/robotwin_hdf5.py` |
| shared action/world Flow-Euler samplers | `src/robonana/sampling.py` |
| FP32 EMA, best-of-eight search, detached TD target | `src/robonana/training/posttraining.py` |
| batch wiring, flow corruption, losses, metrics, checkpoint/resume | `src/robonana/training/robotwin_trainer.py` |
| fixed posttraining contract for both model sizes | `src/robonana/configs/posttrain_config.py` |
| 4B and 800M entry configs | `src/robonana/configs/robotwin_flux2_4b_dino_posttrain.py`, `src/robonana/configs/robotwin_flux2_800m_dino_posttrain.py` |

`q_target_mode="mc_success"` selects pretraining labels from successful
demonstrations. `q_target_mode="td_posttrain"` makes the dataset emit a
placeholder Q; the two posttraining entry configs also enable the trainer path
that replaces it with the EMA TD target before flow noise is applied. Do not
set `td_posttrain` on a pretraining config by itself, because the raw dataset
placeholder is intentionally zero and is not a valid learning target.

### Four-pool replay sampler

The initial RoboTwin dataset is never physically merged with collected replay.
The training dataset is a `ConcatDataset` of four `RoboTwinHDF5Dataset` views,
and every local batch receives deterministic pool quotas:

| pool id | logical pool | nominal batch share | contents |
|---:|---|---:|---|
| 0 | `original_success` | 25% | original successful demonstrations |
| 1 | `collected_success_replay` | 25% | successful policy rollouts from every round |
| 2 | `historical_failure_replay` | 25% | failures with `round_id < current_round` |
| 3 | `latest_failure` | 25% | failures with `round_id == current_round` |

Within each pool the sampler selects task uniformly, then episode uniformly
within that task, then frame uniformly within that episode. This prevents large
tasks or long episodes from silently dominating. Fractional quotas are rounded
with stable largest-remainder allocation. If collected success is empty, its
25% moves to original success. If historical failure is empty, its 25% moves to
latest failure. Any other requested-but-empty pool is a hard error. Advancing
`ROBONANA_COLLECTION_ROUND` reclassifies previous latest failures as historical
without copying or deleting their files.

Each loaded sample preserves both the current frame and the reset-pre final
observation required for timeout bootstrap. The important batch fields are:

| field | meaning |
|---|---|
| `behavior_action` | normalized 48-step action chunk actually executed in the stored trajectory |
| `future_state`, `future_latents`, `future_dino_images` | real observation target at clipped `t_h` |
| `reward`, `success` | direct `-1/0` reward and terminal-success label at `t_h` |
| `reward_h` | real discounted prefix reward `R_t^(delta)` |
| `delta_steps` | count of valid environment transitions, excluding clipped padding/reset |
| `success_terminal_h` | 1 only when `t_h` is the true successful terminal |
| `time_limit_truncated_h` | 1 when a failed rollout reaches its time limit |
| `failure_episode_mask` | selects failure-only best-of-eight action distillation |
| `q_loss_mask` | 1 iff `delta_steps > 0` |

### Failure-only action improvement

Success samples keep the recorded behavior chunk as the Stage-1 action-flow
target. For each historical or latest failure sample, the following search is
performed without gradients:

$$
a_i \sim \pi_\theta(\cdot\mid s_t),\quad i=1,\ldots,8,
$$

$$
q_i = Q_{\bar{\theta}}(s_t,a_i;h=48), \qquad i=1,\ldots,8.
$$

$$
q_{i_{\mathrm{best}}} = \max_{1\leq i\leq 8} q_i, \qquad
a^{\mathrm{pseudo}}=a_{i_{\mathrm{best}}}.
$$

The eight action-noise tensors are independent, while the future-state and Q
noise used by the EMA world sampler is shared across candidates. Reward and
success are direct heads and therefore have no sampling noise.
This common-random-number comparison reduces ranking variance. The existing
state/reward/success/Q-only Stage-2 path is used, so candidate ranking creates no
future-image or DINO tokens and does not invoke the VAE.

The best candidate always becomes the noisy predicted-action track `A` target
for that failure sample. The behavior action is not inserted as a ninth
candidate, and there is no advantage, confidence, or uncertainty gate. The
full-clean `G` track always remains the action actually executed in the replay
trajectory. Consequently:

| prediction/loss | success sample target | failure sample target | conditioning action `G` |
|---|---|---|---|
| predicted action | recorded behavior | best-of-8 pseudo action | recorded behavior |
| future state/image/DINO | recorded future | recorded future | recorded behavior prefix |
| future-state reward/success | recorded one-state reward and success label | recorded one-state reward and success label | recorded behavior prefix |
| current Q | EMA TD target | EMA TD target | recorded behavior prefix |

This separation is deliberate: failed data improves the action generator while
the world and critic targets stay aligned with transitions that actually
occurred.

### EMA TD target

For every sample from all four pools, the trainer replaces the dataset's Q
placeholder with a detached TD target. At `s_(t_h)`, EMA first samples one
48-step continuation action and then samples its Q through the no-image world
path:

$$
a' \sim \pi_{\bar\theta}(\cdot\mid s_{t_h}), \qquad
Q' = Q_{\bar\theta}(s_{t_h},a';h=48).
$$

The target is

$$
y_t^{Q} = R_t^{(\delta)} +
\gamma^{\delta} b_h Q', \qquad
b_h = 1-d_h^{\mathrm{success}}, \qquad \gamma=0.999.
$$

Terminal handling is strict:

| case at `t_h` | `b_h` | Q loss | interpretation |
|---|---:|---:|---|
| ordinary transition | 1 | enabled | bootstrap from EMA continuation |
| successful terminal | 0 | enabled when `delta>0` | target is exactly `reward_h` |
| failed time-limit truncation | 1 | enabled when `delta>0` | bootstrap from stored reset-pre final observation |
| clipped `delta_steps=0` sample | either | disabled | no fabricated zero-length TD constraint |

Only a successful terminal stops bootstrap. A timeout is a truncation, not a
terminal success, and the collector must therefore save the final observation
before reset. The target is computed under `torch.no_grad()` and is independent
of whether a future version supplies MC, TD, or another scalar target to the
same Q flow head.

The full-model EMA is rank-local, stored in FP32, excluded from DDP and the
optimizer, run in eval/no-grad mode with BF16 autocast, and updated only after a
finite, non-skipped optimizer step:

$$
\bar\theta \leftarrow 0.995\,\bar\theta + 0.005\,\theta.
$$

Gradient-accumulation micro-steps do not update EMA. If any rank observes a
non-finite accumulated loss, every rank cancels that optimizer, scheduler, and
EMA step together.

### Joint training objective

Actions, future FLUX latents, future state, Q, and DINO targets use the same
rectified-flow corruption. For clean target `x`, Gaussian noise `epsilon`, and
sampled noise level `sigma`:

$$
x_\sigma=(1-\sigma)x+\sigma\epsilon, \qquad
v^{*}=\epsilon-x.
$$

The corresponding adapter/head predicts `v*`; inference integrates from
`sigma=1` pure noise to `sigma=0` clean data with the shared multi-step Euler
sampler. Q is therefore a scalar flow sample. Reward is a direct scalar
regression target (`-1` nonterminal, `0` successful terminal), and success is a
direct binary logit. Neither direct head receives its target as input. The
weighted training objective is

$$
\mathcal L =
1.0\mathcal L_{image}
+10.0\mathcal L_{action}
+0.4\mathcal L_{state}
+0.01\mathcal L_{reward}
+0.01\mathcal L_{success}
+0.001\mathcal L_Q
+0.1\mathcal L_{DINO}.
$$

The flow terms and direct reward use MSE; success uses binary cross entropy with
logits. The action mask is enabled for both success and failure samples after
the failure pseudo target has been selected; the Q mask excludes only
`delta_steps=0`. No loss is backpropagated through candidate search, EMA
next-action sampling, EMA next-Q sampling, or target construction.

### Interpreting Q during inference

`action_reward_q`, `world_all`, and `world_horizon` return direct `rewards`,
`success_probs`, and sampled `qs` together with the matching `horizons` tensor.
For a supplied action chunk and horizon `h`:

- the reward entry estimates the one-state reward at the selected future state,
  while `success_probs` estimates whether that state is a successful terminal;
- the corresponding Q entry estimates that prefix return plus the learned EMA-policy
  continuation after the predicted `s_(t+h)`;
- larger Q is better because step rewards are non-positive;
- Q must not be interpreted as a probability or added to `reward_h` again—the
  current Q target already contains `reward_h`;
- ranking candidate chunks uses `Q(h=48)` directly.

The optimized `action_reward_q` path queries `h=48` first. If its success
probability is below threshold, it skips horizons 1 through 47 and analytically
accumulates 48 nonterminal rewards. Only when `success_48` is positive does it
query all horizons to locate the first predicted terminal and construct the
discounted accumulated-reward curve.

Before posttraining, Q has the MC-success semantics described in
[What the reward and Q outputs mean](#what-the-reward-and-q-outputs-mean).
After posttraining, it has the bootstrapped behavior-prefix semantics above.
The checkpoint lineage must therefore be recorded when comparing Q values.

### Launch, monitoring, and exact resume

Launch one round from the trained `theta_k` checkpoint:

```bash
export ROBONANA_REPLAY_ROOT=/path/to/separate/rollout-replay
export ROBONANA_POSTTRAIN_CHECKPOINT=/path/to/theta_k/transformer/diffusion_pytorch_model.bin
export ROBONANA_POSTTRAIN_MODEL_CONFIG=/path/to/theta_k/config.json
export ROBONANA_COLLECTION_ROUND=0
export ROBONANA_PROJECT_DIR=$PWD/experiments/robotwin_posttrain_round0

# Canonical pretrained Klein 4B + DINO lineage
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_4b_dino_posttrain.config

# Test/smoke-only 800M + DINO lineage
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_800m_dino_posttrain.config
```

The posttraining W&B namespace reports pool counts, pseudo/success action sample
counts, candidate Q mean/std, best and behavior Q, best-minus-behavior Q, TD
reward/delta/discount/next-Q/target statistics, terminal/timeout/bootstrap
fractions, Q-mask coverage, EMA update count and online distance, plus candidate
search and TD target time/peak memory.

Each checkpoint contains the normal online model, optimizer, scheduler, and
trainer state plus:

```text
ema_model.safetensors  complete FP32 EMA model state
ema_state.json         decay, update count, dtype, and collection round
posttrain_config.json  exact data-mixture/search/TD settings
```

Resume restores all of them. Loading an older online-only RoboNana checkpoint
emits the normal load report and initializes EMA by an exact online copy; it
does not guess or silently synthesize model architecture parameters. The
bounded two-step GPU smoke is:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/smoke_posttraining.py --device cuda:0
```

For the focused hanging-mug experiment, the resumable orchestration script
collects the baseline evaluation and rollout replay in one 50-seed simulator
pass, runs the posttraining round, evaluates the resulting checkpoint, and
writes `comparison.json`. The replay remains outside the initial dataset:

```bash
export ROBONANA_PRETRAIN_CHECKPOINT=$PWD/experiments/robotwin_flux2_4b_dino_reward_success_q_from150k_plus10k/models/checkpoint_epoch_1_step_160000/transformer/diffusion_pytorch_model.bin
export ROBONANA_PRETRAIN_MODEL_CONFIG=$PWD/experiments/robotwin_flux2_4b_dino_reward_success_q_from150k_plus10k/config.json
export ROBONANA_POSTTRAIN_STEPS=1000
bash scripts/run_hanging_mug_posttrain_round.sh
```

The default posttraining run uses physical GPUs 7 and 8 (`CUDA` indices 6 and
7), local batch 4, serial OIDN evaluation on GPU 8, and the training-seen
instruction policy. Completed stages are marked under the output `state/`
directory so a relaunch does not repeat a finished episode. On a fresh round,
`pretrain_eval.done` and `rollout_replay.done` are published together only after
the replay caches are complete. A partially completed episode ledger resumes in
place; run directories produced by the previous two-pass pipeline remain
supported.

## Verification

Executable validation is performed only on `pyromind-west1-58` from a checkout
under `/workspace/hongjia/robonana`:

```bash
bash scripts/verify_remote.sh

# Optional real DINO weight/GPU test
ROBONANA_TEST_REAL_DINO=1 bash scripts/verify_remote.sh
```

Data, caches, checkpoints, outputs, logs, credentials, and upstream source trees
are ignored and must never be committed.
