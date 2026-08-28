# RoboNana

RoboNana is a RoboTwin 2.0 world-action model built by extending FACT's data and
training path with one shared FLUX.2 DiT. The repository keeps upstream FACT and
FLUX.2 implementations external and contains only the adapters, masks, cache
contracts, losses, training hooks, and RoboTwin integration needed by the current
experiment.

## Current experiment

The supported training target is `robonana.configs.robotwin_flux2_800m_dino.config`:

- scratch FLUX.2-shaped DiT: 800,781,312 parameters;
- hidden size 1536, 12 attention heads, 4 double-stream blocks, and 16
  single-stream blocks;
- ordinary BF16 DDP on eight GPUs, without ZeRO-2 or gradient checkpointing;
- 120,000 optimizer steps at global batch size 256 (`32 x 8`);
- FACT RoboTwin-v2: 2,500 Clean episodes plus 25,000 Randomized episodes;
- language, robot state, three-view current image, action, horizon, future state,
  cumulative reward, MC Q, future FLUX-AE image, and a training-only future
  DINOv3 target;
- fixed-horizon W&B visualization at `h = 12, 24, 48` every 1,000 steps.

Training samples `idx_h` uniformly from `1..48`. A target beyond an episode uses
the final valid frame/action, which matches the retained FACT tail-clipping
behavior.

## Shared-DiT design

The exact token order is:

```text
[language | state | current_image | pred_action | gt_action_full_clean |
 idx_h | future_state | reward | Q | future_image_vae | future_image_dino]
```

All segments pass through the same FLUX.2 double-stream and single-stream
blocks. Image, action, future-state, reward, Q, and DINO predictions use separate
output projections; they are not separate transformer backbones. The DINO
branch adds only two projections whose middle dimension follows the selected
shared DiT hidden size:

```text
800M: dino_in/out = Linear(3072, 1536) / Linear(1536, 3072)
4B:   dino_in/out = Linear(3072, 3072) / Linear(3072, 3072)
```

The attention policy is prefix-structured:

| Query | Visible keys |
|---|---|
| language, state, current image | language, state, current image |
| predicted action token `A_t` | clean prefix + full `A_1..A_48` chunk |
| full-clean GT action token `G_t` | clean prefix + `G_1..G_t` |
| horizon | clean prefix + `G_1..G_idx_h` + horizon |
| future state | previous world prefix + future state |
| cumulative reward | previous world prefix + cumulative reward |
| MC Q | previous world prefix + MC Q |
| future FLUX latent | previous world prefix + future FLUX latent |
| future DINO | complete world prefix + future DINO |

The noisy predicted-action track A is bidirectional inside its 48-token chunk,
so diffusion denoises the complete action trajectory jointly. The full-clean
conditioning track G is causal: token `t` cannot read token `t+1`. For every
sample, horizon, future state, reward, Q, future FLUX latent, and future DINO can
read only the first `idx_h` full-clean action tokens.
The mask is constructed dynamically from the batch's `idx_h` tensor. The
predicted-action track remains a sink and is never visible to the GT/world path.
DINO is a final one-way auxiliary sink, so earlier tokens cannot depend on it.
Inference omits the DINO suffix and uses the unchanged action and future-image
samplers.

Let `t_h = min(t + idx_h, T - 1)` and `delta = t_h - t`. `future_state` is
exactly the single robot state at `t_h`; it is not the last state of the
48-action chunk. Successful demonstrations use `gamma=0.999`, reward `-1` at
each non-terminal frame, and reward `0` at the successful terminal frame. The
reward token targets `sum(k=0..delta-1) gamma^k r_(t+k)`. The Q token targets
the full Monte Carlo return from `t`, `sum(k=0..T-1-t) gamma^k r_(t+k)`, so Q is
independent of the sampled horizon for a fixed `t`. Neither target is min-max
normalized or interpreted as time-to-go.

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

The 4B and non-DINO config modules remain only as inheritance layers used by the
current 800M+DINO config. They are not the default experiment entrypoints.

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

The resumable full pipeline builds metadata and Qwen3/FLUX caches, validates
their shapes, waits for eight idle GPUs and valid W&B authentication, then starts
the 120k-step job:

```bash
export ROBONANA_DATASET_ROOT=/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2/RoboTwin
export ROBONANA_FLUX_CHECKPOINT_DIR=$PWD/checkpoints/FLUX.2-klein-base-4B
setsid -f bash scripts/run_full800m_pipeline.sh

cat /data3/hongjia/robonana-jobs/full800m_dino_bs256_120k/status.txt
tail -f /data3/hongjia/robonana-jobs/full800m_dino_bs256_120k/pipeline.log
```

For a bounded cache smoke test, use `--max-tasks` or `--max-episodes` explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python scripts/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "$ROBONANA_DATASET_ROOT" \
  --checkpoint "$ROBONANA_FLUX_CHECKPOINT_DIR" \
  --stage all --max-episodes 1 --batch-size 16 --language-batch-size 1
```

## Training

After the Qwen3 and FLUX cache families validate, the default launcher runs the current
800M+DINO config:

```bash
export ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7
export ROBONANA_BATCH_SIZE=32
export ROBONANA_MAX_STEPS=120000
export ROBONANA_PIXEL_EVAL_INTERVAL=1000
export ROBONANA_CHECKPOINT_INTERVAL=1000
bash scripts/run_robotwin_train.sh
```

The FLUX.2 Klein 4B+DINO config keeps the same dataset, token order, attention
mask, output heads, and loss contract. It uses pretrained Klein 4B, ZeRO-2, no
gradient checkpointing, local micro-batch 16, two accumulation steps
(`16 x 8 x 2 = 256` global batch), and GT-action-only Stage-2 pixel eval every
2,000 optimizer steps. AdamW uses `2e-5` for the pretrained FLUX backbone and
`1e-4` for RoboNana heads, token embeddings, and DINO adapters; both groups
share the same warmup/cosine multiplier:

```bash
export ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7
export ROBONANA_BATCH_SIZE=16
export ROBONANA_MAX_STEPS=120000
export ROBONANA_PIXEL_EVAL_INTERVAL=2000
export ROBONANA_BACKBONE_LR=2e-5
export ROBONANA_ROBOT_LR=1e-4
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_4b_dino.config
```

The joint flow loss weights are image `1.0`, action `10.0`, future state `0.4`,
reward `0.01`, Q `0.001`, and DINO `0.1`. Periodic training visualization performs only
Stage 2: it conditions on the batch's full-clean GT action and samples future
image/state/reward/Q from pure noise with 20-step Flow-Euler. It does not run
Stage-1 action diffusion. The standalone inference modes below are unchanged.

Checkpoint loading is strict. Evaluation must find a complete FACT `config.json`
next to the experiment checkpoint or receive `--model-config`; model size is
never guessed from checkpoint tensor shapes.

## Inference modes

`scripts/inference_server_robotwin.py --inference-mode <mode>` exposes four
strict graphs:

| mode | action source | Stage-2 horizons | future image |
|---|---|---|---|
| `action` | Stage-1 diffusion | none | none |
| `action_reward_q` | Stage-1 diffusion | all `1..T` in one packed pass | omitted |
| `world_all` | request `action_chunk` | all `1..T` in packed horizon batches | FLUX latent + decoded pixels |
| `world_horizon` | request `action_chunk` | request scalar `horizon` | FLUX latent + decoded pixels |

`action_chunk` is the absolute robot-space `[T, action_dim]` chunk; the policy
applies the same delta conversion and z-score normalization used in training.
Packed Stage-2 uses one shared clean causal action track followed by isolated
`[idx_h | future_state_h | reward_h | Q_h | future_image_h]` blocks. A block sees only
`G_1..G_h` and itself, never another horizon block. State/reward/Q-only inference
creates zero future-image tokens and never invokes the VAE decoder. DINO is a
training-only branch and is absent from every inference mode. Image horizons
default to four isolated blocks per forward because packing 48 full FLUX image
grids into one dense-attention sequence is impractical; configure this with
`--stage2-image-horizon-batch-size`.

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

# 2. Stage-1 plus state/reward/Q for every h=1..48; no future VAE/DINO tokens.
"${ROBONANA_SERVER[@]}" --inference-mode action_reward_q --port 8094

# 3. Supplied action chunk plus all state/reward/Q/VAE/pixel horizons.
"${ROBONANA_SERVER[@]}" \
  --inference-mode world_all \
  --stage2-image-horizon-batch-size 4 \
  --vae-decode-batch-size 4 \
  --port 8094

# 4. Supplied action chunk and h plus one state/reward/Q/VAE/pixel prediction.
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

The eight-way launcher audits train/eval instructions, evaluates all 50 tasks,
saves native MP4s, one reward/Q trace per episode, and the decoded Stage-2 future
images. Each 48-action chunk uses one `h=24` reward/Q prediction, which is overlaid on every
executed frame in that chunk.

```bash
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin
export ROBONANA_DATASET_ROOT=/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2/RoboTwin
export ROBONANA_EVAL_GPUS=0,1,2,3,4,5,6,7
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

For success-rate sweeps, auxiliary Stage-2 outputs are unnecessary. The fast
path keeps one FLUX model replica per GPU, runs multiple RoboTwin tasks against
that replica, batches concurrent Stage-1 requests into one model forward, and
still saves the episode MP4s:

```bash
export ROBONANA_EVAL_JOBS_PER_GPU=2
export ROBONANA_EVAL_BATCH_WAIT_MS=100
export ROBONANA_EVAL_AUX_OUTPUTS=0
export ROBONANA_SAPIEN_DENOISER=optix
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

The 100 ms queue window is intentional: real three-view TCP requests fragmented
into 3+5 or 4+4 batches at 6-20 ms, while 100 ms consistently formed full
batches. Start with two jobs per GPU. Increase to four only after a two-worker
OptiX smoke passes without SAPIEN renderer errors. OIDN is process-global in
this runtime and the launcher therefore rejects every multi-process OIDN
configuration. `ROBONANA_EVAL_AUX_OUTPUTS=1`
retains the reward/Q Stage-2 artifact workflow and deliberately requires
`ROBONANA_EVAL_JOBS_PER_GPU=1`.

The server-side RoboTwin 2.0 checkout used for the new official scheduler is
`/data3/hongjia/RoboTwin-official-main-3095469` (official commit `3095469`,
including its pinned XPolicyLab submodule). The old simulator checkout remains
available as a rollback until compatibility validation is complete.

The default renderer denoiser is OptiX for parallel throughput. To match the
official OIDN rendering path, run serially on one GPU:

```bash
export ROBONANA_EVAL_GPUS=2
export ROBONANA_SAPIEN_DENOISER=oidn
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

Blackwell requires the pinned OIDN-compatible environment installed by:

```bash
bash scripts/install_sapien_oidn_blackwell.sh /path/to/robotwin2-oidn233
export ROBOTWIN_CONDA_ENV=/path/to/robotwin2-oidn233
```

## Failure rollout collection

Rollouts are always written outside the initial dataset. The current collector
saves aligned three-view RGB, observed state, actually executed action,
success/failure metadata, Qwen3 context, and FLUX AE cache into a separate HDF5
collection:

```bash
export ROBONANA_INITIAL_DATASET_ROOT="$ROBONANA_DATASET_ROOT"
export ROBONANA_STATS_SOURCE="$ROBONANA_DATASET_ROOT/robonana_norm_stats.json"
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin

TEST_NUM=1 PORT=8095 \
bash scripts/collect_prepare_robotwin_rollouts.sh \
  beat_block_hammer demo_clean step1000_failures 0
```

Failed episodes have `action_loss_mask=0` while retaining their world/reward/Q
supervision. The separate rollout format and collection path are maintained, but
the current 800M+DINO config does not yet mix these episodes into training: the
800M config must first add an explicit LeRobot+HDF5 mixture. The HDF5 loader can
already decode the horizon-selected three-view RGB online for DINO. Do not set
rollout-mixture environment variables for the current DINO run until the
mixture is configured explicitly.

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
