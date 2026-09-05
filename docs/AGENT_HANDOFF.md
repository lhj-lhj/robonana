# RoboNana agent handoff

Last updated: 2026-09-05 (Asia/Shanghai).

Read this file before operating the repository. The current maintained RL
architecture is `mac_mot_v2`; do not reconstruct an older RL path from an old
experiment directory or checkpoint name.

## Authority and safety

| Purpose | Authority |
|---|---|
| Source edits and Git push | `D:\Robotic\robonana` |
| Git remote | `https://github.com/lhj-lhj/robonana`, branch `main` |
| Validation host | `hongjia@208.64.254.190` |
| Canonical server checkout | `/data3/hongjia/robonana` |
| Initial RoboTwin data | `/workspace/datasets/fact-robotwin-v2/RoboTwin` |
| Hanging-mug replay | `/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k` |

Server `_tmp/`, `checkpoint_snapshots/`, `experiments/`, datasets,
rollouts, outputs, logs, and credentials are user artifacts. Never delete,
move, overwrite, or commit them. Inspect process commands and GPU ownership
before launching or stopping anything.

Always derive run status from the live process, logs, status files, checkpoint
contents, and saved experiment `config.json`; directory names are not
evidence.

## Current architecture

There is exactly one FLUX actor/world model. The fixed-horizon sequence is:

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward | success | future_state | future_image_vae]
```

The horizon is always 48. There is no `idx_h`, Value, Q, or DINO token in this
path. World information flows
`reward -> success -> future_state -> future_image_vae`; none of those targets
can read the predicted-action track.

Reward is one query producing 48 binary logits. Success is one endpoint logit.
Value and Q are deterministic scalar MoT experts, each with exactly one learned
query:

- Value reads frozen-FLUX per-layer K/V for
  `[language,state,current_image]`.
- Q reads frozen-FLUX per-layer K/V for
  `[language,state,current_image,clean_action_chunk]`.
- Neither expert uses flow noise or action-expert diffusion.
- Q has no target/EMA copy.
- Only the Value expert has an FP32 target/EMA copy.
- The actor/world FLUX is completely frozen during critic optimization.

The expert block and initialization structure is adapted from ImageWAM at
pinned commit `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`; exact source links
are in `src/robonana/models/flux2_scalar_expert.py`. The MAC target/source
links are beside the target code in `src/robonana/sampling.py` and
`src/robonana/training/posttraining.py`.

## Training contract

Every round uses two serial training jobs.

### Phase 1: `world_policy`

Train the single FLUX actor/world model on real replay:

- successful data: action BC and every world loss;
- failed data: zero action-loss weight, but reward/success/future-state/
  future-image losses remain enabled.

No full-model EMA is created.

### Phase 2: `critic`

Load the exact phase-1 checkpoint, freeze FLUX and all actor/world adapters,
then train only `value_expert` and `q_expert`. Each batch performs one fresh
H=1 imagined rollout. Online Q selects the highest-valued sampled action chunk.
The detached targets are:

```text
V_target = R_chunk + gamma^48 * nonterminal * target_V(next)
Q_target = R_chunk + gamma^48 * nonterminal * online_V(next)
```

After a finite, non-skipped optimizer step, update only target Value. The
checkpoint stores `target_value_expert.safetensors` and
`value_ema_state.json`.

### Environment feedback loop

Environment inference samples M policy chunks, scores them with online Q, and
executes `argmax Q`. Use M=32 for the selected-policy evaluation/collection
path. The rollout stores all candidate Q values and selection metadata.
Selected successful trajectories enter the next round's action BC; failed
trajectories remain world-model training data with action loss disabled.

The maintained orchestration entrypoint is:

```bash
bash scripts/run_hanging_mug_mac_round.sh
```

It runs `world_policy`, then `critic`, then same-seed M=1 and M=32
evaluation, selected-policy collection, replay-cache refresh, and comparison
publication. Its `state/` markers make completed stages resumable.

## Initialization and lineage

The first round loads:

```text
/data3/hongjia/robonana/experiments/
robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k/
models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin
```

Use the `config.json` in the same experiment directory. Migration keeps
compatible FLUX/image/action/state tensors, skips all old project heads, and
initializes the two new slim experts from FLUX using the ImageWAM resize policy.
Later rounds must exact-load the prior `mac_mot_v2` checkpoint and saved
configuration. The round launcher also forwards the prior critic's
`target_value_expert.safetensors` and `value_ema_state.json`; do not drop these
when continuing the Value target across rounds.

`legacy_v1` remains only for first-stage variable-`idx_h` pretraining and
strict legacy checkpoint/inference compatibility. Do not mix its token
semantics or scalar-flow heads into v2.

## Maintained implementation map

| Responsibility | File |
|---|---|
| actor/world model and critic K/V prefill | `src/robonana/models/mac_flux2_fact.py` |
| one-query slim Value/Q experts | `src/robonana/models/flux2_scalar_expert.py` |
| actor/world and critic masks | `src/robonana/models/attention_mask.py` |
| 120k migration and trainable surfaces | `src/robonana/models/pretrained.py` |
| one-step imagination and Q rejection | `src/robonana/sampling.py` |
| Value-only EMA | `src/robonana/training/posttraining.py` |
| two-phase losses/checkpoint hooks | `src/robonana/training/robotwin_trainer.py` |
| phase/config contract | `src/robonana/configs/posttrain_config.py` |
| complete hanging-mug round | `scripts/run_hanging_mug_mac_round.sh` |
| real-checkpoint architecture smoke | `scripts/validate_mac_mot_v2_checkpoint.py` |

## Verification evidence

On 2026-09-05, the temporary 190 validation checkout passed:

```text
151 passed, 1 skipped
```

A real Klein-4B migration from the step-120000 checkpoint reported:

```text
architecture_version: mac_mot_v2
loaded_parameter_tensors: 153
skipped_legacy_tensors: 9
expert_hidden_dim: 1024
Value expert parameters: 563355904
Q expert parameters: 563355904
FLUX trainable parameters in critic phase: 0
target Q: absent
EMA FLUX: absent
```

The real BF16 GPU forward/backward smoke on physical GPU 6 also confirmed that
both experts receive gradients and no frozen-FLUX parameter receives a
gradient. Re-run these checks after any model, attention, migration, or trainer
change:

```bash
python -m pytest -q
python scripts/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <120k diffusion_pytorch_model.bin> \
  --model-config <120k config.json> \
  --device cuda:0 \
  --smoke-forward
```

## Startup audit

Before operational work on 190:

```bash
cd /data3/hongjia/robonana
git status --short
git rev-parse HEAD
git rev-parse origin/main
readlink -f /home/hongjia/robonana
nvidia-smi
ps -eo user,pid,lstart,cmd --sort=lstart | grep -E 'robonana|robotwin' | grep -v grep
```

For any selected experiment, record
`(host, repo realpath, Git HEAD, experiment dir, saved config, checkpoint,
PID/command, W&B id, dataset roots/indexes)` before diagnosing or resuming it.
