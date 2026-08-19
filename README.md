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

Every 200 optimizer steps, the trainer evaluates each selected current frame at
fixed horizons `idx_h = 12, 24, 48` after backward and optimizer completion.
Every rank evaluates a different local current frame, lazily loads the frozen
FP32 FLUX.2 AE, and locally decodes its current/GT/predicted images. Each AE is
immediately removed from GPU; compact uint8 pixels are gathered to rank 0 for
CPU composition and upload as one eight-row W&B panel. Evaluation mirrors
inference: it first samples action
from pure noise, feeds the resulting clean action into the teacher-forcing
track, then jointly samples future image/state/value from pure noise with
20-step Flow-Euler. Set `ROBONANA_NUM_INFERENCE_STEPS` to use a different
shared eval/inference step count.

Ordinary training batches do not carry fixed-horizon eval targets. They retain
only a scalar sample index; after periodic pure-noise sampling completes, the
three GT future latents are loaded lazily for W&B comparison only.

The launcher tees combined stdout/stderr to a timestamped file under
`$ROBONANA_PROJECT_DIR/logs`. In addition to the normal 1000-step checkpoint
interval, step 100 is saved by default; override it with a comma-separated
`ROBONANA_EARLY_CHECKPOINT_STEPS` value.

See [docs/INHERITANCE.md](docs/INHERITANCE.md) for the exact reuse boundary.

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
