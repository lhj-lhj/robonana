# robonana

Minimal implementation scaffold for a RoboTwin 2.0 world-action model that:

- reuses FACT's dataset/transform contract;
- subclasses the official `flux2.model.Flux2` backbone;
- sends image, action, state, horizon, and value tokens through the same FLUX.2 DiT blocks;
- adds only token adapters, output heads, per-segment timestep modulation, and an explicit attention mask.

## Upstream source trees

The repository intentionally does not copy FACT or FLUX.2 source files. Make them importable:

```bash
git clone https://github.com/Bariona/FACT.git third_party/FACT
git clone https://github.com/black-forest-labs/flux2.git third_party/flux2
export PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src:$PYTHONPATH"
```

## Verification

All verification is run on `pyromind-west1-58` under `/workspace/hongjia/robonana`:

```bash
bash scripts/verify_remote.sh
```

## W&B tracking

Training uploads to the `robonana` project in the
`hongjia-liu-aalto-university` entity by default. Authenticate once on each
training host with `wandb login`; never place the API key in Git, Notion, a
launcher, or a committed config file.

## Real RoboTwin training

The production path now uses FACT's `Trainer`, `DefaultCollator`, sampler
registry, optimizer/scheduler builders, Accelerate/DeepSpeed wrapping,
checkpointing, and W&B scalar logging. RoboNana adds only a raw-HDF5 dataset
adapter and the FLUX-specific forward step.

Generate the portable episode index and FACT-compatible normalization stats
once after the FLUX/Qwen caches are complete:

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src" \
  .venv/bin/python scripts/compute_robotwin_metadata.py \
  --dataset-root /workspace/datasets/RoboTwin/hf_dataset
```

Launch full shared-DiT training. `ROBONANA_GPU_IDS` defaults to `0,2,5,7`,
batch size defaults to one per GPU, and multi-GPU execution reuses FACT's
DeepSpeed ZeRO-2 config.

```bash
ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7 \
ROBONANA_BATCH_SIZE=16 \
  bash scripts/run_robotwin_train.sh
```

Every 100 optimizer steps, the trainer evaluates each selected current frame at
fixed horizons `idx_h = 12, 24, 48` after backward and optimizer completion.
Every rank evaluates a different local current frame, lazily loads the frozen
FP32 FLUX.2 AE, and locally decodes its current/GT/predicted images. Each AE is
immediately removed from GPU; compact uint8 pixels are gathered to rank 0 for
CPU composition and upload as one eight-row W&B panel. Evaluation mirrors
inference: it first samples action
from pure noise, feeds the resulting clean action into the teacher-forcing
track, then jointly samples future image/state/value from pure noise with
20-step Flow-Euler. A second panel reuses the same future noise but conditions
the world stage on the dataset GT action, isolating world-model alignment from
action-generation error. W&B keys are `eval/fixed_horizon_grid` and
`eval/fixed_horizon_gt_action_grid`. Set `ROBONANA_NUM_INFERENCE_STEPS` to use
a different shared eval/inference step count.

Ordinary training batches do not carry fixed-horizon eval targets. They retain
only a scalar sample index; after periodic pure-noise sampling completes, the
three GT future latents are loaded lazily for W&B comparison only.

Training horizons use a rollout-aligned mixture by default: 50% of samples use
`idx_h = 24`, while the remaining 50% draw uniformly from `1..48`. Override the
anchor and mixture probability with `ROBONANA_ROLLOUT_HORIZON` and
`ROBONANA_ROLLOUT_HORIZON_PROB`. Current-frame inputs remain clean cached FLUX
latents; no online pixel augmentation or VAE encoding is performed.

The launcher tees combined stdout/stderr to a timestamped file under
`$ROBONANA_PROJECT_DIR/logs`. In addition to the normal 1000-step checkpoint
interval, step 100 is saved by default; override it with a comma-separated
`ROBONANA_EARLY_CHECKPOINT_STEPS` value.

See [docs/INHERITANCE.md](docs/INHERITANCE.md) for the exact reuse boundary.

### Scratch 200M BF16 DDP experiment

`robonana.configs.robotwin_flux2_small200m.config` keeps the full original
50-task/2500-episode dataset and its language, token order, inputs, outputs,
episode-uniform, action-chunk, tail-clip, and horizon sampling behavior. It
initializes an approximately 200M-parameter FLUX.2-shaped DiT from scratch,
disables both model-native gradient checkpointing and FACT activation
checkpointing, and launches ordinary BF16 `MULTI_GPU` DDP without DeepSpeed.

```bash
ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7 \
ROBONANA_BATCH_SIZE=32 \
ROBONANA_MAX_STEPS=10000 \
ROBONANA_PROJECT_DIR="$PWD/experiments/robotwin_flux2_small200m" \
  bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_small200m.config
```

The small config defaults to learning rate `1e-4` with 500 warmup steps. Use
`ROBONANA_LR` and `ROBONANA_WARMUP_STEPS` to override them.

### Full RoboTwin-v2 scratch 800M experiment

`robonana.configs.robotwin_flux2_800m.config` reads FACT's released LeRobot-v2
layout directly: 50 Clean tasks with 50 episodes each plus 50 Randomized tasks
with 500 episodes each. It reuses the same RoboNana token order, action chunk,
tail clipping, language conditioning, horizon mixture, heads, and losses. The
FLUX-shaped DiT has 791,333,376 parameters (`hidden_size=1536`, 12 heads, 4
double-stream blocks, 16 single-stream blocks), is initialized from scratch,
and runs ordinary BF16 DDP without gradient checkpointing.

The resumable background pipeline computes full-dataset norm stats/index,
per-episode Qwen3 contexts, per-frame FLUX AE tokens, validates all 27,500
episodes, waits until all eight GPUs are free, then starts 120k steps at global
batch size 256 (32 samples per GPU):

```bash
setsid -f bash scripts/run_full800m_pipeline.sh
cat /data3/hongjia/robonana-jobs/full800m_bs256_120k/status.txt
tail -f /data3/hongjia/robonana-jobs/full800m_bs256_120k/pipeline.log
```

The source archive is not duplicated when its size and Hugging Face LFS SHA256
already match. Preprocessing uses GPUs 1-7 by default so a pre-existing GPU 0
job is left untouched; training waits for all eight GPUs rather than evicting
another process.

## RoboTwin FLUX caches

Raw RoboTwin HDF5 tasks are converted without changing FACT's three-view
layout. Qwen3 is run once per task and every RGB frame is encoded once; the
training loader selects current/future tokens using `current_index` and
`idx_h`.

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src" \
  .venv/bin/python scripts/preprocess_robotwin_flux.py \
  --dataset-root /workspace/datasets/RoboTwin/hf_dataset \
  --checkpoint checkpoints/FLUX.2-klein-base-4B \
  --stage language

PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src" \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/preprocess_robotwin_flux.py \
  --dataset-root /workspace/datasets/RoboTwin/hf_dataset \
  --checkpoint checkpoints/FLUX.2-klein-base-4B \
  --stage images --batch-size 16
```

On west1-58, the resumable four-GPU launcher defaults to the currently agreed
physical GPUs `0,2,5,7`:

```bash
setsid -f bash scripts/run_robotwin_flux_cache.sh
tail -f logs/preprocess_robotwin_flux_full.log
```

## RoboTwin simulation evaluation

RoboNana keeps FACT's TCP client and RoboTwin `eval_policy.py`; only the online
FLUX/Qwen encoder and checkpoint adapter are new.  RoboTwin commit `2eeec32`
expects CuRobo's pre-refactor API, so pin the checkout created by RoboTwin's
installer to the newest compatible tag once:

```bash
git -C /workspace/hongjia/RoboTwin/envs/curobo switch --detach v0.7.8
/workspace/.conda/envs/robotwin2/bin/python -m pip install --no-build-isolation \
  -e /workspace/hongjia/RoboTwin/envs/curobo
```

Start the step-100 policy server with the DiT on physical GPU 6 and FLUX AE on
physical GPU 7:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src:$PWD/third_party/flux2_official/src" \
.venv/bin/python scripts/inference_server_robotwin.py \
  --checkpoint experiments/robotwin_flux2_h24mix_bs12_10k_20260819/models/checkpoint_epoch_1_step_100/transformer/diffusion_pytorch_model.bin \
  --flux-checkpoint-dir checkpoints/FLUX.2-klein-base-4B \
  --stats-path /workspace/datasets/RoboTwin/hf_dataset/robonana_norm_stats.json \
  --model-device cuda:0 --vae-device cuda:1 --port 8094
```

In a second shell, run one RoboTwin episode on physical GPU 7.  Do not force
`VK_ICD_FILENAMES`; SAPIEN's automatic ICD selection is required on west1-58.

```bash
CUDA_VISIBLE_DEVICES=7 \
XDG_RUNTIME_DIR=/tmp/robonana_robotwin_eval_8094 \
PYTHONPATH="$PWD/src" \
FACT_CONDA_ENV="$PWD/.venv" \
ROBOTWIN_PATH=/workspace/hongjia/RoboTwin \
ROBOTWIN_CONDA_ENV=/workspace/.conda/envs/robotwin2 \
POLICY_NAME=robonana_robotwin.adapter \
PORT=8094 TEST_NUM=1 EXECUTE_ACTIONS_PER_PLAN=48 \
SERVER_TIMEOUT_MS=600000 SERVER_WAIT_SECONDS=600 EVAL_VIDEO_LOG=0 \
bash third_party/FACT/evaluation/robotwin/launch_client.sh \
  beat_block_hammer demo_clean step100 0
```

The inference server resolves model size from the exact training metadata. It
discovers the FACT `config.json` above the checkpoint, or accepts an explicit
`--model-config /path/to/config.json`. The config must contain complete
`models.params`, action/state/value dimensions, and `max_horizon`; missing or
partial metadata is a hard error and is never inferred from checkpoint tensors.

For the canonical 50-task evaluation, use the eight-way launcher. It starts one
800M inference server and one RoboTwin client per GPU, assigns the 50 tasks
round-robin, enables native MP4 recording, and saves one value trace per episode:

```bash
export ROBONANA_TRAINED_CHECKPOINT="$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin"
export ROBONANA_EVAL_GPUS=0,1,2,3,4,5,6,7
bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

The launcher first audits all 27,500 training episodes. Their metadata prompt
must match RoboTwin's `seen` template family and exactly match the prompt stored
beside the Qwen3 cache. Evaluation therefore uses `instruction_type: seen`, not
RoboTwin's default `unseen`. Each plan runs Stage-2 world sampling at `h=24`,
denormalizes its scalar value, stores it in `values_per_plan` inside the episode
NPZ trace, and overlays the same `chunk/h/value` label on every native video
frame executed from that 48-action chunk. The run fails if it does not produce
all 50 task results, `50 * test_num` MP4s, and the same number of value traces.

## Separate rollout collection and retraining

Policy-generated data is never written into the initial RoboTwin root.  Each
collection has its own dataset root:

```text
/workspace/hongjia/robonana_rollouts/<collection>/
  robonana_collection.json
  robonana_index.json
  robonana_norm_stats.json
  robonana_ready.json
  <task>/robonana_rollout/
    data/episode0.hdf5
    instructions/episode0.json
    metadata/episode0.json
    flux_cache/language/episode_000000.pt
    flux_cache/latents/episode_000000.pt
```

One command runs RoboTwin, records aligned three-view RGB/state/executed action,
stops its temporary inference server, builds the episode index, copies the
initial normalization contract, and generates episode-level Qwen3 plus FLUX AE
caches:

```bash
export ROBONANA_TRAINED_CHECKPOINT="$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin"
TEST_NUM=1 PORT=8095 \
ROBONANA_SERVER_GPU_IDS=6,7 \
ROBONANA_CLIENT_GPU_ID=7 \
ROBONANA_PREPARE_GPU_ID=7 \
bash scripts/collect_prepare_robotwin_rollouts.sh \
  beat_block_hammer demo_clean step1000_failures 0
```

The writer rejects any rollout root nested inside
`/workspace/datasets/RoboTwin/hf_dataset`.  It stores observed state in
`joint_action/vector` and the action actually sent to RoboTwin separately in
`policy_action/vector`.  Failed episodes carry `success=false`, use the FACT
failure value penalty, and return `action_loss_mask=0` in the dataset.

To continue training, restart the same experiment directory with the collection
enabled.  The checkpoint resumes normally, while the initial and rollout files
remain in separate roots and are joined only by the sampler:

```bash
export ROBONANA_PROJECT_DIR="$PWD/experiments/<run>"
export ROBONANA_ROLLOUT_DATASET_ROOT=/workspace/hongjia/robonana_rollouts/step1000_failures
export ROBONANA_ROLLOUT_DATASET_WEIGHT=1.0
bash scripts/run_robotwin_train.sh
```

`ROBONANA_ROLLOUT_DATASET_WEIGHT` is the rollout collection's sampling weight
relative to the initial dataset's fixed weight of `1.0`.  Restart training after
adding episodes so DataLoader workers rebuild the collection index.
