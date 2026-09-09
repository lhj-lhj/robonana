# Maintained code map (2026-09-08)

This map and README describe current behavior. Dated experiment reports record
historical behavior; do not reconstruct legacy branches from those reports.

## Single runtime contract

- Precision (2026-09-09): FACT BF16 for FLUX/online Q/V in train and eval.
  Value EMA stores/updates FP32 and evaluates with BF16 autocast on BF16 cache.
  Loss/returns and VAE BN inverse normalization stay FP32; encoding caches and
  Qwen are unchanged.
  No FP32-only execution wrapper or global BF16 reduction override remains.
- Architecture: `models/mac_flux2_fact.py` implements the fixed-48 MAC model; FACT
  and FLUX supply upstream blocks. Keep deterministic MoT Q/V and Value-only
  EMA; no architecture/attention/loss changes were made by this cleanup.
- Training: `training/robotwin_trainer.py`, config `robotwin_flux2_4b_mac`.
  `critic_continuation` edits a copy of a saved configuration, preserving the
  saved run. Every new pool uses A, including when the saved run used B.
- State/action normalization: **only** `normalization.py` and its
  `A_STATS_PATH`. Dataset, online loader and config guards reject other paths.
  CLI stats flags can name A, but are not alternate normalization modes.
- Image inputs: `image_pipeline.py` + `encoding.py`; original LeRobot and HDF5
  share these adapters with single/batched online inference. Rebuild historical
  image caches into certified `latents_v2` before starting training.
- Replay preparation: `scripts/prepare_robotwin_rollouts.py` builds an episode
  index through `data/stats.py::write_robotwin_replay_index`, references A and
  generates image/language caches. It does not refit or copy normalization.
- Collection: RLinf/RoboTwin pool management in `sim/collection_pool.py`,
  existing inference servers/transports, and lossless HDF5 rollout writer.
- Selected-action world reports: `inference/selected_world.py` still calls
  `_decode_stage2_image`; keep this helper even though its old docstring said
  legacy. The vectorized decoder remains shared.

## Deliberate cleanup

Removed after checking source/script/test callers:

1. Unreachable `if False` legacy world response branch and its output assembly
   in `robotwin_policy.py`, plus `_sample_world`, called only by that branch.
   The second reference audit also removed the disabled variable-horizon reward
   curve/query chain, its `_sample_stage2*` wrappers, external-action inverse
   adapter and unused horizon-batch CLI flag. Current reports use
   `selected_world.predict_selected_world -> sample_mac_world`, not this chain.
2. `scripts/diagnose_robotwin_batch_numerics.py`, the one-off investigation of
   the former batch-dependent VAE. Its old solo/batched VAE comparison no
   longer tests the removed pipeline. Keep the recorded results in outputs
   and `COLLECTION_POOL.md`; use `verify_image_pipeline.py` for current parity
   and `benchmark_robotwin_inference_batch.py` for current action batching.
3. HDF5 replay statistics fitting/writing functions. Tests of the removed
   refitting route were replaced with tests that replay indexing preserves
   episode metadata and cannot create/overwrite normalization files.
4. Legacy `sample_flux2_world`, `sample_world_flow`, `sample_two_stage_flow`
   and flow-Q result containers; the action sampler now accepts MAC only.
   Trainer forward dispatch is exclusively `world_policy` or `critic`.
5. The unreachable non-MAC forward, joint flow-Q/DINO loss, online DINO encoder,
   DINO dataset image loading, and their obsolete tests/dependency. Removed the
   old periodic multi-horizon pixel-eval chain and its dataset accessors too.
   Current `sample_mac_world`, shared VAE decoding and selected-world reports
   are retained. Continuation strips removed configuration keys from a copy;
   original saved run configurations are never rewritten.

The shared `Flux2FACTModel` base still supplies modules and block helpers used
by `MacFlux2FACTModel`; it is not an independently supported training route.
This cleanup does not rename inherited parameters or migrate checkpoint weights.

Retained intentionally: source LeRobot metadata/statistics provenance utility,
pilot/fit probes, batch/cache benchmarks, both required socket transports,
RoboTwin/SAPIEN runtime fixes, and checkpoint/distributed-safety tests. These
still have callers or explicit operational uses; absence from a default run
alone does not make a module unused. No datasets, caches, experiment outputs,
credentials or checkpoints were deleted. Removed tracked code is recoverable
through Git history, not maintained as another runtime version.

## Verification scope

Use the complete pytest suite on 190, compile all Python, check shell syntax,
and search for removed symbols/callers. Tests must exercise normalization A
selection, rejection of B, source-config immutability during continuation,
replay metadata preservation, and selected-world report paths. Existing Q/V,
attention-cache and distributed tests remain required. Do not start a training
run just to validate cleanup or overwrite old caches to make a check pass.
