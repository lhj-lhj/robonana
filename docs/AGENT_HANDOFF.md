# RoboNana agent handoff and operational log

Last verified: 2026-09-05 (Asia/Shanghai).
Fixed-48 MAC implementation and 190 validation are on branch `main`; always
re-check local, origin, and server HEAD before quoting a commit.

This document is the first-read handoff for an agent with no conversation
history. It records the current research objective, authoritative locations,
run/data lineage, and the checks required before making a claim. It is not a
substitute for live inspection: paths, commits, processes, checkpoints, and run
status must be verified at the start of every task.

## Non-negotiable evidence rules

1. Do not claim that a job is running, finished, failed, or resumable from its
   directory name. Check the process command, launcher/train log, status file,
   checkpoint contents, and W&B run.
2. Do not claim which dataset produced an image from its appearance. Resolve
   `pool_id -> dataset child -> EpisodeRecord -> source/cache path`; log these
   fields when possible.
3. Do not infer an experiment's configuration from a Python base config. Read
   the experiment's saved `config.json`, because environment variables and
   inheritance override defaults.
4. Do not silently infer a model size from checkpoint tensors. A complete
   `model_config.json` is required; missing configuration is an error.
5. Never expose or commit W&B/Hugging Face tokens. Use the already-authenticated
   CLI/session.
6. Do not interrupt unrelated GPU processes. Inspect PID, command line, GPU and
   owner before stopping anything.
7. Server checkouts are for execution and validation. Make focused source edits
   and Git commits locally, push from local, then pull on the server.
8. Generated datasets, caches, rollouts, checkpoints, videos and credentials do
   not belong in Git.

## Authoritative locations

| Purpose | Authority |
|---|---|
| Source editing and Git push | `D:\Robotic\robonana` on Windows |
| Git remote | `https://github.com/lhj-lhj/robonana`, branch `main` |
| Execution and validation host | `hongjia@208.64.254.190` |
| Canonical server checkout | `/data3/hongjia/robonana` |
| Convenience symlink | `/home/hongjia/robonana -> /data3/hongjia/robonana` |
| Initial FACT RoboTwin-v2 data | `/workspace/datasets/fact-robotwin-v2/RoboTwin` |
| Current hanging-mug replay | `/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k` |
| Local downloaded artifacts | `D:\Robotic\robonana\outputs\downloads` |

Do **not** use west1-58 or an old migration/validation worktree unless the user
explicitly changes the authority. Do not assume local and server HEAD match:

On 190 the `/workspace/datasets/fact-robotwin-v2` symlink currently resolves
into `/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2`. That target
is retained **dataset storage**, not the source-code authority. Do not delete it
as an "old repository". Source execution remains `/data3/hongjia/robonana`.

```powershell
cd D:\Robotic\robonana
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
```

```bash
ssh hongjia@208.64.254.190
cd /data3/hongjia/robonana
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
```

## Research objective and current model contract

The current objective is the fixed-48 `mac_v1` RL path. RoboNana still reuses
FACT's data/training bus and the official FLUX.2 Klein 4B shared DiT; there is
no MoT or separate action/world transformer. The maintained MAC config disables
DINO.

Current MAC token order:

```text
[language | state | current_image_vae | Value | pred_action |
 gt_action_full_clean | Q | reward | success | future_state | future_image_vae]
```

There is no `idx_h`: all action chunks and future targets use 48 steps. Value
can read only language/state/current image. Q can additionally read the full
clean action chunk. Reward has the same inputs as Q; success additionally reads
reward; future state additionally reads success; future image additionally
reads future state. World targets cannot read Value, predicted action, or Q.

Value and Q are deterministic scalar heads. One reward query emits 48 binary
logits rather than regressing a drifting scalar chunk return. The first
implementation performs one H=1 on-policy learned-world rollout per update,
uses M=8 candidate chunks during training, and bootstraps from an always-on FP32
EMA Value. Environment inference samples M=32 BC chunks and executes the
deterministic-Q argmax.

Selected successful rollouts re-enter action BC on the next round. Both success
and failure rollouts train future image/state/reward/success, while failures
have zero action-BC weight. Candidate Q values and the selected candidate are
stored in rollout HDF5.

The first MAC round loads the immutable step-120000 checkpoint from:

```text
/data3/hongjia/robonana/experiments/robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k/models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin
```

Its exact source config is the experiment-level `config.json`. Migration keeps
the official FLUX backbone and compatible action/state/image adapters, but
skips all obsolete horizon/segment/DINO/Value and other project heads. Later
rounds use exact `trained` loading from the preceding `mac_v1` checkpoint and
its config.

The legacy `mc_posttrain` experiment below is retained as historical evidence;
it is not the current token/Q contract and its checkpoint must not be loaded as
MAC.

## Current hanging-mug critic-calibration experiment

### Corrected hanging-mug-only run (completed 2026-09-03)

```text
experiment: /data3/hongjia/robonana/experiments/hanging_mug_mc_posttrain_550success_50replay_from160k_4k
W&B: https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/wcgu55w3
initialization: step160000 reward/success/Q checkpoint
GPUs: 6,7
per-GPU batch: 16
gradient accumulation: 2
global batch: 64
steps: 4000
pixel eval: every 2000 steps
```

Its physical data contract was checked before launch:

```text
original_success: Clean/hanging_mug 50 + Randomized/hanging_mug 500
collected_success_replay: hanging_mug 5
latest_failure: hanging_mug 45
outcome sampling: 50% success / 50% failure
failure action loss: disabled; all other enabled losses remain active
```

At launch, commit `406f80d` was checked out on 190. Training completed at step
4000 with launcher exit code 0 and produced the complete checkpoint at:

```text
/data3/hongjia/robonana/experiments/hanging_mug_mc_posttrain_550success_50replay_from160k_4k/models/checkpoint_epoch_2_step_4000
```

The subsequent 25-episode `demo_clean` evaluation completed with 9 successes
and 16 failures, for a 36% success rate and no infrastructure errors. It wrote
25 HDF5 episodes plus a complete index and normalization statistics under:

```text
evaluation: /data3/hongjia/robonana/outputs/hanging_mug_mc_posttrain_550success_step4000_eval25_capture
rollouts: /data3/hongjia/robonana_rollouts/hanging_mug_mc_posttrain_550success_step4000_eval25_capture
```

The local-only return-overlay pipeline also completed with 125 videos, 125
telemetry JSON files, and 125 manifest rows. The groups are 50 `expert_clean`,
50 `collected_pre_5of50`, and 25 `mc_posttrain_550success_eval25` episodes:

```text
/data3/hongjia/robonana/outputs/hanging_mug_mc_posttrain_550success_step4000_overlay125_20260903
```

Its final status is `complete_local_only`; these artifacts were not uploaded to
Hugging Face. Always inspect the live process and output state before reporting
current progress or launching another run.

### Previous mixed-index run

Experiment directory:

```text
/data3/hongjia/robonana/experiments/hanging_mug_mc_posttrain_100traj_from160k_4k
```

W&B run:

```text
https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/cc7ebtpf
```

Intended contract:

- initialization: hanging-mug-compatible step-160000 checkpoint;
- two GPUs, physical GPUs 6 and 7;
- per-GPU batch 16, gradient accumulation 2, global batch 64;
- 4,000 optimizer steps;
- fixed-horizon pixel monitoring at steps 2,000 and 4,000;
- data outcome balance: 50% success and 50% failure;
- physical pools: 50 original clean successes plus the 50 collected trajectories
  containing 5 successes and 45 failures;
- failure action loss is zero; all other enabled heads still train.

The saved experiment config is the source of truth:

```bash
jq .dataloaders.train \
  /data3/hongjia/robonana/experiments/hanging_mug_mc_posttrain_100traj_from160k_4k/config.json
```

This run was trained before commit `c8c931f`, so its `original_success` pool
contains all tasks from the global LeRobot index, while its collected-success
and failure replay pools contain hanging_mug only. The user explicitly accepted
this mixed-data lineage on 2026-09-03 and authorized continued use of the
step-4000 checkpoint for hanging_mug evaluation. Do not describe it as a pure
hanging_mug run, but do not reject the checkpoint for this reason. The
`robonana-mc4k-eval-overlay150` automation is active again.

Check status without changing the run:

```bash
cd /data3/hongjia/robonana
exp=experiments/hanging_mug_mc_posttrain_100traj_from160k_4k
ps -eo pid,lstart,cmd | grep -F "$exp" | grep -v grep
tail -n 120 "$exp/launcher.log"
tail -n 120 "$exp"/logs/*.log
find "$exp/models" -maxdepth 3 -type f | sort | tail -n 80
```

## Data lineage

### Initial expert pool

```text
root: /workspace/datasets/fact-robotwin-v2/RoboTwin
format: LeRobot-v2
requested task_globs: Clean/hanging_mug
filter: success
pool_name: original_success
```

### Collected replay

```text
root: /data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k
format: RoboNana rollout HDF5
task_glob: **/robonana_rollout
index: robonana_index.json
contents: 50 episodes = 5 success + 45 failure
policy_version: reward_success_q_160k
```

The previously returned overlay set contains 150 videos (50 expert, 50
collected-pre, 50 posttrain-eval). The user visually verified that these videos
do not contain randomized backgrounds. Do not attribute the step-2000 W&B
image to randomized replay without new evidence.

## Resolved step-2000 pixel-eval incident

Observed symptom: the step-2000 W&B fixed-horizon image showed scenes from
other RoboTwin tasks; step 4000 happened to show hanging-mug scenes.

Root cause: when a global LeRobot `robonana_index.json` existed,
`load_lerobot_episode_records()` loaded every indexed episode and ignored its
`task_globs` argument. Therefore the posttraining config's
`task_globs=("Clean/hanging_mug",)` did not actually restrict the original
success child. Step 2000 sampled other tasks; step 4000 sampled hanging_mug by
chance. The renderer and clean simulator configuration were not the cause.

Required invariant after the fix:

```text
every EpisodeRecord loaded from a global index must have task_dir in the set
resolved from the configured task_globs
```

Regression test:

```text
tests/test_robotwin_lerobot.py::test_global_lerobot_index_still_respects_task_globs
```

Validation evidence on 190 at commit `c8c931f`:

```text
python -m pytest -q tests/test_robotwin_lerobot.py tests/test_posttrain_config.py
6 passed in 5.29s

global LeRobot index: 50 distinct task names
configured Clean/hanging_mug result: 50 records
filtered task-name set: ['hanging_mug']
```

Pixel eval currently stages the first sample on each rank and later maps
`pool_id` back to the matching dataset child for lazy GT-latent loading. When
debugging future images, inspect both the staged current latent and the lazy GT
source. A future observability improvement should log `pool_name`, task,
episode, frame, and cache path in the W&B caption; never guess these fields.

## Main entrypoints

| Operation | Maintained entrypoint |
|---|---|
| preprocess initial LeRobot data | `scripts/preprocess_robotwin_lerobot_flux.py` |
| train | `scripts/run_robotwin_train.sh` -> `scripts/train_robotwin.py` |
| inference server | `scripts/inference_server_robotwin.py` |
| isolated RoboTwin task eval | `scripts/eval_robotwin_task_isolated.py` |
| full-task parallel eval | `scripts/eval_robotwin_all_tasks_parallel.sh` |
| collect and preprocess replay | `scripts/collect_prepare_robotwin_rollouts.sh` |
| fixed-48 MAC train/eval/selected-policy round | `scripts/run_hanging_mug_mac_round.sh` |
| iterative hanging-mug round | `scripts/run_hanging_mug_posttrain_round.sh` |
| annotate recorded Q/reward overlays | `scripts/annotate_recorded_robotwin_returns.py` |
| package the three 50-video groups | `scripts/run_hanging_mug_overlay150_pipeline.sh` |

Before launching, inspect the script and run `--help` where supported. Reuse
these paths; do not create a parallel trainer, dataset, evaluator, or uploader.

## Minimum new-agent startup audit

Run these checks before answering operational questions:

```bash
cd /data3/hongjia/robonana
git status --short
git rev-parse HEAD
readlink -f /home/hongjia/robonana
nvidia-smi
ps -eo user,pid,lstart,cmd --sort=lstart | grep -E 'robonana|robotwin' | grep -v grep
find experiments -maxdepth 2 -name config.json -printf '%T@ %p\n' | sort -nr | head
```

For a selected experiment, record this tuple before diagnosing it:

```text
(host, repo_realpath, git_HEAD, experiment_dir, config.json,
 checkpoint/model_config.json, PID+command, W&B run id, dataset roots/indexes)
```

If any element is missing, say exactly what is unknown. Do not fill the gap
from memory or from a similarly named run.

## Updating this log

Update this file whenever one of these changes materially:

- canonical server or checkout;
- default model/token/mask contract;
- active experiment and checkpoint;
- dataset/replay lineage;
- maintained launch/eval/collection entrypoint;
- known correctness issue or its fix.

Every update should include a date and evidence path. Keep secrets, transient
progress chatter, and unverifiable conclusions out of this file.
