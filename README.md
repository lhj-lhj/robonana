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

The checkpoint-backed BS=1 graph smoke test loads the official FLUX.2 Klein
Base 4B DiT checkpoint (not WAN2.2):

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> PYTHONPATH="$PWD/src:$PWD/third_party/flux2/src" \
  .venv/bin/python scripts/smoke_train.py \
  --checkpoint checkpoints/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors \
  --batch-size 1 --train-mode adapters --memory-limit-gib 14
```

The script checks free memory before creating a CUDA tensor. `adapters` is a
low-memory wiring test; the intended shared-DiT experiment uses `--train-mode
full` and is refused when there is not enough room for weights, gradients, and
AdamW state.

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
