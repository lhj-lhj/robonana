# RoboNana

For a concise, evidence-oriented handoff to a new agent, read
[docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) before running commands.

RoboNana is a RoboTwin 2.0 world-action model built by extending FACT's data and
training path with one shared FLUX.2 DiT. The repository keeps upstream FACT and
FLUX.2 implementations external and contains only the adapters, masks, cache
contracts, losses, training hooks, and RoboTwin integration needed by the current
experiment.

## Current fixed-48 MAC experiment (`mac_mot_v2`)

The maintained RL entrypoint is
`robonana.configs.robotwin_flux2_4b_mac_from120k.config`. It uses a fixed
48-step action chunk, one H=1 learned-world rollout per critic batch,
deterministic scalar Value/Q MoT experts, one query that emits 48 binary reward
logits, a target/EMA copy of the Value expert only, M=8 candidates during
training, and M=32 Q-rejection sampling during environment inference.

The exact MAC token order is:

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward | success | future_state | future_image_vae]
```

There is no `idx_h` token in `mac_mot_v2`; `future_state` and
`future_image_vae` always refer to the clipped `t+48` observation. Value and Q
are not tokens in this sequence. Their one-query experts read detached
per-layer K/V from the frozen FLUX backbone: Value uses
`[language,state,current_image]`, while Q additionally uses the complete clean
48-action chunk. See
[docs/INHERITANCE.md](docs/INHERITANCE.md) for the exact non-leaking attention
dependencies.

The first MAC round warm-starts from the immutable 120k run. Its loader keeps
the official FLUX.2 backbone and shape-compatible action/state/image adapters,
but explicitly skips old horizon/segment/DINO/Value/Q and other project heads.
Later MAC rounds use exact `trained` loading from the prior MAC run and its
saved `config.json`.

## Legacy pretraining lineage (`legacy_v1`)

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

## Legacy shared-DiT design

This section documents the `legacy_v1` lineage retained for checkpoint
compatibility and historical inference. It is not the current MAC layout.

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

### Reward, success, Value, and Q semantics

In the retained `legacy_v1` pretraining path, `idx_h` is still sampled from
`1..48`; its reward/success/Q targets keep their historical meanings for
checkpoint compatibility.

The maintained `mac_mot_v2` path always uses a complete 48-step chunk. Its
world model predicts 48 Bernoulli reward logits and one endpoint-success logit.
For logit `ell_j`, the expected step reward is
`-1 + sigmoid(ell_j)`, and the discounted chunk reward is

```text
R_chunk = sum(j=0..47, gamma^j * (-1 + sigmoid(ell_j)))
```

Value and Q are deterministic scalar regressors, not flow variables and not
probabilities. Value sees only language, state, and the current FLUX-AE image.
Q sees the same context plus the complete clean 48-step action chunk. Larger Q
is better because the step rewards are non-positive.

Critic training follows the original one-rollout MAC target split:

```text
V target = R_chunk + gamma^48 * nonterminal * target_value(next_state)
Q target = R_chunk + gamma^48 * nonterminal * online_value(next_state)
```

Both targets are stop-gradient. Only the Value expert has a target/EMA copy;
there is no target Q and no EMA FLUX.

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
| fixed-48 MAC RL | `robonana.configs.robotwin_flux2_4b_mac_from120k.config` | DINO disabled; deterministic V/Q plus binary chunk reward |
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

`scripts/inference_server_robotwin.py --inference-mode <mode>` exposes the
legacy first-stage/world graphs plus the maintained MAC rejection graph:

| mode | action source | Stage-2 horizons | future image |
|---|---|---|---|
| `action` | Stage-1 diffusion | none | none |
| `action_q_rejection` | M Stage-1 samples, online-Q argmax | none | none |
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

## Fixed-48 MAC training and selected-policy loop

One training round has two serial phases. The launcher never trains both
surfaces with the same optimizer:

1. `world_policy`: train the single FLUX actor/world model on real replay.
   Successful samples train action BC. Failed samples have zero action-loss
   weight. Both outcomes always train reward, success, future-state, and
   future-image losses.
2. `critic`: freeze every FLUX actor/world parameter, perform exactly one
   fresh H=1 learned-world rollout, and train only the online Value and Q
   experts. After a successful optimizer step, update the FP32 target Value
   expert. There is no target Q and no EMA copy of FLUX.

The actor/world sequence is:

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward | success | future_state | future_image_vae]
```

The learned cascade is
`reward -> success -> future_state -> future_image_vae`. Each Value/Q expert
owns one learned query and follows ImageWAM's slim-FLUX cached-attention
structure. Per-layer K/V comes from the frozen main FLUX. Value caches
`[language,state,current_image]`; Q additionally caches the complete clean
action chunk. The expert implementation and initialization cite the pinned
ImageWAM source revision directly in
`src/robonana/models/flux2_scalar_expert.py`.

For one imagined selected action, let

$$
\hat r_j=-1+\operatorname{sigmoid}(\ell_j),\qquad
\hat R_{48}=\sum_{j=0}^{47}\gamma^j\hat r_j,
$$

and let the hard nonterminal mask be zero when the predicted endpoint success
probability is at least 0.5. The detached regression targets are

$$
y^V=\hat R_{48}+\gamma^{48}m\,
V_{\bar\phi}(\hat s_{t+48}),\qquad
y^Q=\hat R_{48}+\gamma^{48}m\,
V_{\phi}(\hat s_{t+48}).
$$

This matches the original MAC implementation: online Q is trained against the
online Value bootstrap, and only Value has a Polyak target. Code comments link
the corresponding MAC source lines.

The first round migrates from the immutable step-120000 checkpoint:

```text
experiments/robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k/
  models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin
```

The loader reconstructs that checkpoint from its saved experiment
`config.json`, whitelists FLUX plus compatible action/state/image adapters,
skips every old project head, and initializes each slim expert from FLUX using
ImageWAM's axis-wise interpolation and alpha scaling. Later rounds load the
preceding `mac_mot_v2` checkpoint exactly.

Run a complete resumable hanging-mug round with:

```bash
export ROBONANA_REPLAY_ROOT=/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k
export ROBONANA_COLLECTION_ROUND=0
export ROBONANA_PROJECT_DIR=$PWD/experiments/hanging_mug_mac_round0
export ROBONANA_MAC_WORLD_POLICY_STEPS=10000
export ROBONANA_MAC_CRITIC_STEPS=10000
bash scripts/run_hanging_mug_mac_round.sh
```

The script runs the two training phases in order, evaluates the resulting
critic checkpoint with M=1 and M=32 on the same seeds, collects the M=32
selected-policy trajectories, refreshes their caches, and writes
`comparison.json`. Stage completion is recorded under `state/`, so relaunch
does not repeat completed work. For later rounds it also carries the preceding
critic's `target_value_expert.safetensors` through the intervening frozen-expert
world phase, preserving the Value Polyak trajectory.

Environment inference uses online-Q rejection sampling:

```bash
export ROBONANA_INFERENCE_MODE=action_q_rejection
export ROBONANA_REJECTION_CANDIDATE_COUNT=32
```

The policy samples M independent BC action chunks, evaluates deterministic
online Q for every candidate, and executes the argmax. Candidate Q values,
selected index, selected Q, and margin are stored in rollout HDF5. On the next
round, selected successful trajectories participate in action BC; selected
failures continue to train the world model but never the action loss.

The removed full-model-EMA, scalar-Q flow, `td_posttrain`, and
`mc_posttrain` launch paths are intentionally unsupported. Only the
first-stage `legacy_v1` variable-`idx_h` path remains for checkpoint and
pretraining compatibility.

## Verification

Executable validation is performed on `hongjia@208.64.254.190` from the
canonical checkout `/data3/hongjia/robonana`:

```bash
bash scripts/verify_remote.sh

# Real 120k migration and frozen-FLUX critic gradient test
python scripts/validate_mac_mot_v2_checkpoint.py --checkpoint <120k-bin> \
  --model-config <120k-config.json> --device cuda:0 --smoke-forward
```

Data, caches, checkpoints, outputs, logs, credentials, and upstream source trees
are ignored and must never be committed.
