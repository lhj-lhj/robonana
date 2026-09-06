# RoboNana

RoboNana is a RoboTwin 2.0 world-action project built around one shared FLUX.2
DiT. The actively maintained reinforcement-learning path is `mac_mot_v2`:

```text
one online FLUX actor/world model
one online deterministic Value expert
one FP32 EMA target Value expert
one online deterministic Q expert
```

There is no EMA copy of FLUX and no target Q. The old variable-`idx_h`
architecture is retained only for loading the 120k pretrained checkpoint and
for explicitly requested legacy pretraining/inference. All legacy details are collected at
the end of this README.

For a shorter operational handoff, see
[docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md). For the exact upstream reuse
boundary, see [docs/INHERITANCE.md](docs/INHERITANCE.md).
The fixed-48 tail policy and distributed checkpoint/nonfinite safety checks
are documented in [the 190 validation report](docs/MAC_TRAINING_SAFETY_VALIDATION_20260906.md).

Documentation map:

- [Current contract](#current-implementation-at-a-glance)
- [Model architecture](#model-architecture)
- [Replay data](#real-replay-data)
- [Phase 1: policy and world model](#phase-1-policy-and-learned-world-model)
- [Phase 2: MAC critics and equations](#phase-2-mac-critic-training)
- [Checkpoint lifecycle](#weight-initialization-and-checkpoint-lifecycle)
- [Environment feedback loop](#environment-policy-and-selected-data-feedback-loop)
- [Run, cache, and evaluation commands](#running-one-complete-hanging-mug-round)
- [Legacy compatibility](#legacy-compatibility-legacy_v1)

## Current implementation at a glance

The maintained configuration is:

```text
robonana.configs.robotwin_flux2_4b_mac_from120k.config
```

Its fixed contract is:

| item | current value |
|---|---|
| architecture | `mac_mot_v2` |
| action-chunk horizon | 48 environment steps |
| imagined rollout horizon | one 48-step chunk (`H=1`) |
| phase-1 trainable parameters | FLUX actor/world side only |
| phase-2 trainable parameters | online Value and online Q experts only |
| training-time action candidates | `M=8` by default |
| environment action candidates | `M=32` by default |
| reward prediction | one query, 48 independent binary logits |
| success prediction | one endpoint terminate logit |
| Value input | language, state, current FLUX-AE image |
| Q input | Value inputs plus the complete clean 48-action chunk |
| Value target network | Value expert only, FP32 Polyak EMA |
| Q target network | none |
| DINO | disabled in `mac_mot_v2` |

One complete round is strictly serial:

```text
real replay
   |
   v
phase 1: train policy + world model
   |
   v
phase 2: freeze FLUX, train Value + Q with one imagined chunk
   |
   v
same-seed M=1 and M=32 environment evaluation
   |
   v
append M=32 selected-policy trajectories to replay
   |
   v
next round
```

## Model architecture

### Actor/world token sequence

The `mac_mot_v2` actor/world sequence is exactly:

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward | success | future_state | future_image_vae]
```

The short names used below are:

```text
C  = [language, state, current_image_vae]
A  = noisy predicted 48-action chunk
G  = clean behavior/selected 48-action chunk
R  = one reward query that emits 48 logits
U  = one endpoint-success/terminate query
S' = future state at the clipped t+48 frame
I' = future FLUX-AE image tokens at the clipped t+48 frame
```

There is no `idx_h`, Value, or Q token in this sequence. The dataset still
provides a `horizon_idx` field for loader compatibility, but the MAC trainer
requires every value to equal 48 and the model does not embed it.

The actor/world attention dependencies are:

| query | visible keys |
|---|---|
| `C` | `C` |
| `A` | `C`, complete `A` chunk |
| `G` | `C`, complete `G` chunk |
| `R` | `C`, `G`, `R` |
| `U` | `C`, `G`, `R`, `U` |
| `S'` | `C`, `G`, `R`, `U`, `S'` |
| `I'` | `C`, `G`, `R`, `U`, `S'`, `I'` |

Consequently:

- the policy denoises all 48 action tokens jointly;
- all 48 clean actions are available to every world target;
- the learned world cascade is `reward -> success -> future_state -> future_image`;
- no world target can read the noisy policy output `A`;
- no future target can leak backward into an earlier target.

The implementation lives in
`src/robonana/models/mac_flux2_fact.py` and
`src/robonana/models/attention_mask.py`.

### Deterministic Value and Q experts

Value and Q are separate MoT-style scalar experts, not actor/world tokens and
not flow-matching variables. Each expert owns exactly one learned query and the
same number of double/single layers as FLUX. Its hidden width defaults to 1024,
while the attention head dimension remains compatible with the 4B FLUX
backbone.

For every FLUX layer, the frozen main branch produces detached K/V. The expert
computes its own query K/V, concatenates the main K/V and expert K/V, applies
mixed attention, and updates only its own query stream:

```text
frozen FLUX prefix --per-layer K/V--+
                                     +--> expert query block --> scalar
expert query -----------own K/V-----+
```

The main FLUX branch never reads expert K/V. Critic prefix computation runs
under `torch.no_grad()`, and every cached tensor is detached before the expert
uses it. Therefore critic loss gradients reach the online Value/Q experts but
cannot reach FLUX.

Value receives:

```text
V(language, state, current_image_vae)
```

Q receives:

```text
Q(language, state, current_image_vae, clean_action_chunk[1:48])
```

Both produce one deterministic normalized-return scalar. Larger Q is better
because the default environment rewards are non-positive.

The frozen Q prefix uses the same asymmetric graph as the world model:
`C -> C`, `G -> C,G`. L/S/I never read a candidate action. Each observation's
per-layer C K/V are computed once and reused across candidates, policy Euler
steps, and Q scoring. Candidate clean-action K/V are stored separately and
joined with C only for the current layer/group. Each Q query sees only its
own candidate. Joint critic-loss forwards also share C between Value and Q.

The slim block math and resize policy are adapted from
[ImageWAM commit `5d4a341`](https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a);
the pinned file-and-line references are also kept beside the adapted code in
`src/robonana/models/flux2_scalar_expert.py`.

## Real replay data

Every MAC sample contains:

```text
language
normalized current state
current three-view FLUX-AE image tokens
normalized executed/behavior action chunk [48, action_dim]
48 binary reward labels and their validity mask
endpoint success/terminate label
normalized future state at min(t+48, episode_end)
future FLUX-AE image tokens at min(t+48, episode_end)
episode success/failure and collection lineage metadata
```

The action target is the actually executed policy action, not a newly sampled
action. Arm joints use the same delta conversion as inference, after which
state and action are normalized with `robonana_norm_stats.json`.

### Reward and terminate labels

For a real transition with `delta_steps <= 48`, the binary reward labels mean:

```text
label 0 -> non-goal step reward -1
label 1 -> absorbing post-success reward 0
```

If the episode succeeds within the chunk, the observed pre-terminal positions
are class 0 and the terminal suffix is class 1. With T real actions and T+1
observations, success windows start at 0..T-1 and may contain up to 47 absorbing
padding steps. Their future state/image repeats the successful final observation;
padding actions are excluded from BC. Failure windows start only at 0..T-48,
inclusive: every window has 48 real actions, and the last window ends at the
actual final observation. No incomplete failure window is sampled or padded.

The success target is one terminate label for the clipped `t+48` endpoint. A
failure time limit is never converted into a success terminal.

### Four replay pools

Each phase samples the same four physical pools with default weights 0.25 each:

| pool | content | action BC? | world losses? |
|---|---|---|---|
| `original_success` | original clean hanging-mug successes | yes | yes |
| `collected_success_replay` | successful selected-policy rollouts | yes | yes |
| `historical_failure_replay` | failures from earlier collection rounds | no | yes |
| `latest_failure` | failures from the current collection round | no | yes |

If a replay-success pool is empty, its probability moves to original success.
If historical failure is empty in the first round, its probability moves to
latest failure.

## Phase 1: policy and learned world model

Phase 1 uses only real replay. It freezes both experts and trains the one shared
FLUX actor/world model.

### Flow-matching targets

For clean target `x`, Gaussian noise `epsilon`, and sampled noise level
`sigma in [0,1]`, training constructs:

$$
x_\sigma=(1-\sigma)x+\sigma\epsilon,
\qquad
v^\star=\epsilon-x.
$$

The action chunk receives its own noise level `sigma_a`. Future state and
future image share a separately sampled world noise level `sigma_w`. The model
regresses their velocity targets with MSE.

### Phase-1 losses

Let `b_i` be 1 for a successful episode and 0 for a failed episode. The policy
loss is success-only behavior cloning:

$$
\mathcal L_A=
\frac{\sum_i b_i\lVert\hat v^A_i-v^{A\star}_i\rVert_2^2}
     {\max(\sum_i b_i,\epsilon)}.
$$

Failures therefore cannot move the policy backward, but they still train every
world target. The remaining losses are:

$$
\begin{aligned}
\mathcal L_I &= \operatorname{MSE}(\hat v^I,v^{I\star}),\\
\mathcal L_S &= \operatorname{MSE}(\hat v^S,v^{S\star}),\\
\mathcal L_R &= \operatorname{masked\_BCEWithLogits}(\ell^R,y^R),\\
\mathcal L_U &= \operatorname{BCEWithLogits}(\ell^U,y^U).
\end{aligned}
$$

The current weighted objective is:

$$
\mathcal L_{\text{phase1}}
=1.0\mathcal L_I
+10.0\mathcal L_A
+0.4\mathcal L_S
+0.1\mathcal L_R
+0.1\mathcal L_U.
$$

Phase 1 has no Value loss, no Q loss, no target Value model, and no EMA FLUX.

## Phase 2: MAC critic training

Phase 2 loads the phase-1 checkpoint, freezes all FLUX actor/world parameters,
and creates a fresh one-chunk imagined transition for every critic batch. Only
the online Value and online Q experts are placed in the optimizer.

### Step 1: on-policy action selection

The policy independently samples `M` complete action chunks from Gaussian
noise with Flow-Euler integration. The default is `M=8` during critic training.

For state input

$$
x_t=(l,s_t,i_t),
$$

candidate selection is:

$$
a_t^{(m)}\sim\pi_\omega(\cdot\mid x_t),
\qquad
m^\star=\arg\max_{m\in\{1,\ldots,M\}}
Q_\theta(x_t,a_t^{(m)}),
$$

$$
\hat a_t=a_t^{(m^\star)}.
$$

Only the online Q expert participates in selection. Target Value is never used
to choose an environment or imagined action.

### Step 2: one learned-world rollout

The selected clean action chunk conditions the frozen world model:

$$
(\hat s_{t+48},\hat i_{t+48},\ell^R_{0:47},\ell^U)
\sim p_\omega(\cdot\mid x_t,\hat a_t).
$$

Future state and future image are denoised from fresh Gaussian noise. Reward
and success are direct logits evaluated in the final world-model call. The
attention mask still restricts them to their declared prefix: reward cannot
read success/future state/future image, and success cannot read future
state/future image. Exactly one 48-step chunk is generated; the implementation
does not recursively imagine a second chunk.

The complete candidate selection and world rollout run without gradients.

### Step 3: convert reward logits to a chunk return

For reward logit `ell_j`, default non-goal reward `r_ng=-1`, and goal/absorbing
reward `r_g=0`, the expected step reward is:

$$
\hat r_j
=r_{ng}+\operatorname{sigmoid}(\ell_j)(r_g-r_{ng})
=-1+\operatorname{sigmoid}(\ell_j).
$$

The discounted 48-step chunk return is:

$$
\hat R_{48}=\sum_{j=0}^{47}\gamma^j\hat r_j.
$$

Success is converted into a detached hard terminate decision:

$$
d=\mathbf 1[\operatorname{sigmoid}(\ell^U)\ge 0.5],
\qquad
m=1-d.
$$

There is no soft terminal probability in the bootstrap term.

### Step 4: MAC Value and Q targets

The original MAC formulation uses action-chunk length `n`, an imagined horizon
of `H` chunks, and samples `k` uniformly from `0..H-1`:

$$
\mathcal L^V(\phi)=\mathbb E\left[
\left(
V_\phi(\hat s_{t+kn})
-\sum_{i=k}^{H-1}\gamma^{(i-k)n}\hat r_{t+in}
-\gamma^{(H-k)n}V_{\bar\phi}(\hat s_{t+Hn})
\right)^2
\right],
$$

$$
\mathcal L^Q(\theta)=\mathbb E\left[
\left(
Q_\theta(s_t,\hat a_t)
-\hat r_t
-\gamma^n\operatorname{stopgrad}(V_\phi(\hat s_{t+n}))
\right)^2
\right].
$$

See the [MAC project page](https://kwanyoungpark.github.io/MAC/) and the
[reference implementation](https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L191-L217).
RoboNana fixes `n=48`, `H=1`, and therefore `k=0`; its `hat r_t` is the
discounted 48-step return `hat R_48` defined above.

The experts actually emit normalized scalars. Let `tilde V_phi` and
`tilde Q_theta` denote those direct outputs, let `c` be the fixed return scale,
and define environment-return-unit values by:

$$
V_\phi=c\tilde V_\phi,
\qquad
Q_\theta=c\tilde Q_\theta.
$$

With `tilde V_bar_phi` denoting the EMA expert output, the detached raw-return
targets used by RoboNana are:

$$
y^V
=\hat R_{48}+\gamma^{48}m\,
c\tilde V_{\bar\phi}(l,\hat s_{t+48},\hat i_{t+48}),
$$

$$
y^Q
=\hat R_{48}+\gamma^{48}m\,
c\tilde V_{\phi}(l,\hat s_{t+48},\hat i_{t+48}).
$$

This is the deliberate MAC split:

- Value bootstraps from target/EMA Value;
- Q bootstraps from stop-gradient online Value;
- there is no target Q.

With `c=ROBONANA_MAC_RETURN_SCALE` (default 1000), the critic losses compare
the direct normalized expert outputs against normalized targets:

$$
\mathcal L_V=\operatorname{MSE}
\left(\tilde V_\phi(x_t),\operatorname{stopgrad}(y^V/c)\right),
$$

$$
\mathcal L_Q=\operatorname{MSE}
\left(\tilde Q_\theta(x_t,\hat a_t),\operatorname{stopgrad}(y^Q/c)\right),
$$

$$
\mathcal L_{\text{phase2}}=\mathcal L_V+\mathcal L_Q.
$$

The final differentiable Value/Q forward uses fresh detached FLUX caches. A
non-finite microstep is detected with a SUM of bad flags across ranks. Every
rank raises before backward/optimizer/scheduler/EMA and aborts the run; do not
continue an unfinished DDP/ZeRO accumulation. Resume from a complete checkpoint.
ZeRO saves include frozen parameters so its module payload supports strict
resume, independently of the separate full transformer export.

### Value EMA update

After a successful optimizer step:

$$
\bar\phi\leftarrow
\tau\bar\phi+(1-\tau)\phi,
\qquad \tau=0.995.
$$

The target Value expert is stored and updated in FP32. Its forward pass uses
BF16 autocast with the BF16 frozen-FLUX cache.

## Weight initialization and checkpoint lifecycle

### First MAC round

Round 0 starts from the immutable legacy step-120000 checkpoint:

```text
experiments/robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k/
  models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin
```

The migration loader reconstructs the source from its saved `config.json` and:

- loads the FLUX backbone exactly;
- loads shape-compatible action, state, image, and final-image projections;
- skips old horizon/segment/DINO/Value/Q/reward/success project heads;
- constructs the two new deterministic experts;
- initializes their slim blocks with the pinned ImageWAM copy/resize policy;
- keeps both learned queries and both complete scalar heads newly initialized,
  including head AdaLN modulation; FLUX `final_layer.*` is not copied into them.

### Later MAC rounds

Round `r>0` exact-loads the preceding critic checkpoint, including its trained
online Value and online Q experts. During the new phase 1, Value and Q remain
frozen while FLUX changes.

At the start of the new critic phase:

```text
online Value(r,start) = online Value(r-1,end)
online Q(r,start)     = online Q(r-1,end)
target Value(r,start) = exact copy of online Value(r,start)
EMA update_count      = 0
```

The previous round's EMA target is intentionally not carried across the new
phase-1 representation change.

If the same critic run is resumed, its own saved target Value and EMA update
count are restored instead. A critic resume checkpoint missing either EMA file
is rejected rather than silently rebuilding the target.

### Checkpoint contents

A phase-1 checkpoint contains the full online model, including the frozen
online Value/Q parameters inherited from the input checkpoint. It does not
contain a target Value because phase 1 has none.

A critic checkpoint contains:

```text
transformer/diffusion_pytorch_model.bin  full online FLUX + Value + Q model
target_value_expert.safetensors          FP32 target Value only
value_ema_state.json                     decay, update count, collection round
posttrain_config.json                    exact MAC phase/configuration
optimizer/scheduler/trainer state        FACT/Accelerate resume state
```

There is no EMA FLUX file and no target-Q file.

## Environment policy and selected-data feedback loop

Live `mac_mot_v2` inference supports:

| mode | behavior |
|---|---|
| `action` | sample one BC action chunk and execute it |
| `action_q_rejection` | sample `M` chunks, score online Q, execute argmax |

The maintained environment mode is `action_q_rejection` with `M=32`:

```bash
export ROBONANA_INFERENCE_MODE=action_q_rejection
export ROBONANA_REJECTION_CANDIDATE_COUNT=32
export ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE=8
```

For one observation, the current implementation:

1. encodes language, normalized state, and the three current RGB views;
2. creates `M` independently seeded Gaussian action noises;
3. prefills L/S/I once at batch size B, then denoises candidate groups using
   this cache across every Euler step (default group size 8);
4. re-encodes each final clean action chunk with the same L/S/I cache, then
   evaluates its deterministic online Q query without computing Value;
5. executes the highest-Q 48-action chunk;
6. records every candidate Q, selected index, selected Q, and Q margin.

Target Value is not loaded by the inference server and cannot affect action
selection.

The selected-policy collector stores aligned RGB, observed state, actually
executed action, success/failure metadata, reset-pre final observation,
`transition_valid`, `round_id`, `policy_checkpoint`, and `policy_version`.
It then builds Qwen3/FLUX caches and appends the episodes to the cumulative
replay root.

In the next phase 1:

```text
selected success -> action BC + every world loss
selected failure -> every world loss, zero action BC weight
```

This closes the policy-improvement loop without directly differentiating the
policy through Q.

## Running one complete hanging-mug round

The canonical round launcher is resumable and writes stage markers under
`$ROBONANA_MAC_RUN_ROOT/state`:

```text
world_policy.done
critic.done
m1_eval.done
m32_collection.done
```

Round 0 example:

```bash
cd /data3/hongjia/robonana

export ROBONANA_REPLAY_ROOT=/data3/hongjia/robonana_rollouts/hanging_mug_round0_from160k
export ROBONANA_COLLECTION_ROUND=0
export ROBONANA_PROJECT_DIR=$PWD/experiments/hanging_mug_mac_round0
export ROBONANA_MAC_WORLD_POLICY_STEPS=10000
export ROBONANA_MAC_CRITIC_STEPS=10000
export ROBONANA_GPU_IDS=6,7

bash scripts/run_hanging_mug_mac_round.sh
```

The launcher executes:

```text
world_policy training
-> critic training
-> same-seed M=1 evaluation
-> same-seed M=32 selected-policy evaluation and collection
-> replay cache preparation
-> comparison.json
```

For a later round, explicitly provide the preceding critic's online checkpoint
and saved configuration:

```bash
export ROBONANA_COLLECTION_ROUND=1
export ROBONANA_MAC_INITIALIZATION=trained
export ROBONANA_MAC_SOURCE_CHECKPOINT=/path/to/round0/critic/checkpoint/transformer/diffusion_pytorch_model.bin
export ROBONANA_MAC_SOURCE_CONFIG=/path/to/round0/critic/config.json
export ROBONANA_PROJECT_DIR=$PWD/experiments/hanging_mug_mac_round1

bash scripts/run_hanging_mug_mac_round.sh
```

Do not pass a previous round's `target_value_expert.safetensors` into a new
round. That file is only for resuming the critic run that created it.

## Environment and offline caches

The canonical server checkout is:

```text
hongjia@208.64.254.190:/data3/hongjia/robonana
```

The maintained Python environment is currently:

```text
/data3/hongjia/conda/envs/robonana/bin/python
```

For a fresh installation, make FACT and FLUX.2 importable:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[train,preprocess,dev]'

git clone https://github.com/Bariona/FACT.git third_party/FACT
git clone https://github.com/black-forest-labs/flux2.git third_party/flux2

export PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
```

Place the official FLUX.2 Klein components at
`checkpoints/FLUX.2-klein-base-4B` or set
`ROBONANA_FLUX_CHECKPOINT_DIR`. The directory must include the local Qwen3 text
encoder and FLUX AE.

The current full dataset root is:

```text
/data3/hongjia/robonana-migration/datasets/fact-robotwin-v2/RoboTwin
```

Per episode, preprocessing writes:

```text
flux_cache/language/episode_NNNNNN.pt  BF16 [512,7680]
flux_cache/latents/episode_NNNNNN.pt   BF16 [T,288,128]
```

The 288 image tokens represent the same three-view `384 x 192` composite used
by training and inference.

Build and validate caches with:

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

Credentials and upstream code are external. Never put Hugging Face or W&B
tokens in Git.

## RoboTwin evaluation and collection

### FACT-compatible isolated-GPU path

The maintained launcher keeps policy inference and SAPIEN/OIDN on disjoint
physical GPU pools. By default, policy requests execute all 48 actions before
sampling the next chunk.

```bash
export ROBONANA_TRAINED_CHECKPOINT=$PWD/experiments/<run>/models/<checkpoint>/transformer/diffusion_pytorch_model.bin
export ROBONANA_MODEL_CONFIG=$PWD/experiments/<run>/config.json
export ROBONANA_DATASET_ROOT=/workspace/datasets/fact-robotwin-v2/RoboTwin
export ROBONANA_EVAL_SERVER_GPUS=0,1,2,3
export ROBONANA_EVAL_SIM_GPUS=4,5,6,7
export ROBONANA_EVAL_JOBS_PER_GPU=1
export ROBONANA_ROBOTWIN_STATIC_CAMERAS=head_camera
export ROBONANA_INFERENCE_MODE=action_q_rejection
export ROBONANA_REJECTION_CANDIDATE_COUNT=32

bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

Each run writes per-task `results.csv`, aggregate `summary.txt`, MP4 manifests,
worker logs, and append-only episode/attempt ledgers. Infrastructure errors are
reported as `ERROR` and are not counted as policy failures.

To evaluate and immediately create trainable replay:

```bash
export ROBONANA_INITIAL_DATASET_ROOT="$ROBONANA_DATASET_ROOT"
export ROBONANA_STATS_SOURCE="$ROBONANA_DATASET_ROOT/robonana_norm_stats.json"
export ROBONANA_COLLECTION_ROUND=1
export ROBONANA_POLICY_VERSION=mac_round0_m32

TEST_NUM=50 PORT=8095 \
bash scripts/collect_prepare_robotwin_rollouts.sh \
  hanging_mug demo_clean hanging_mug_selected_round1 0
```

### Official RoboTwin/XPolicyLab batch path

The official batch adapter converts observations as follows:

```text
cam_head.color        -> observation.images.cam_high
cam_left_wrist.color  -> observation.images.cam_left_wrist
cam_right_wrist.color -> observation.images.cam_right_wrist

state = [left_arm_joint_state, left_ee_joint_state,
         right_arm_joint_state, right_ee_joint_state]
```

One persistent TCP connection owns one simulator observation stream. Concurrent
requests are combined by `BatchedRoboNanaRobotWinPolicy.inference_batch`.

Start the policy server on a GPU not used by SAPIEN:

```bash
python scripts/inference_server_robotwin_xpolicylab.py \
  --checkpoint "$ROBONANA_TRAINED_CHECKPOINT" \
  --model-config "$ROBONANA_MODEL_CONFIG" \
  --flux-checkpoint-dir "$ROBONANA_FLUX_CHECKPOINT_DIR" \
  --stats-path "$ROBONANA_DATASET_ROOT/robonana_norm_stats.json" \
  --xpolicylab-root "$ROBOTWIN_ROOT/XPolicyLab" \
  --model-device cuda:0 --vae-device cuda:0 \
  --max-batch-size 7 --max-batch-wait-ms 100 --port 8094
```

Then run the official evaluator on disjoint simulator GPUs:

```bash
cd "$ROBOTWIN_ROOT"
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

The H100 reference validation used official RoboTwin commit
`30954692d06ba7e89f7a6b76064f4062c488fa81` without modifying its source.

## Repository map

```text
src/robonana/models/mac_flux2_fact.py       current actor/world model and critic caches
src/robonana/models/flux2_scalar_expert.py  one-query Value/Q experts
src/robonana/models/attention_mask.py       actor/world and critic masks
src/robonana/models/pretrained.py            strict 120k migration and trained loading
src/robonana/data/robotwin_hdf5.py           fixed-48 targets and replay lineage
src/robonana/configs/posttrain_config.py     MAC phase/data/EMA contract
src/robonana/sampling.py                     Flow-Euler, Q rejection, H=1 imagination
src/robonana/training/robotwin_trainer.py    losses, freezing, EMA, checkpoint hooks
src/robonana/inference/                      RoboTwin policies and dynamic batching
scripts/run_hanging_mug_mac_round.sh         complete serial round
scripts/collect_prepare_robotwin_rollouts.sh selected-policy replay collection
tests/                                       unit and integration contracts
```

## Verification and current open items

Run tests on the 190 validation host:

```bash
cd /data3/hongjia/robonana
/data3/hongjia/conda/envs/robonana/bin/python -m pytest -q
```

Audit a real step-120000 checkpoint and frozen-FLUX critic backward:

```bash
/data3/hongjia/conda/envs/robonana/bin/python \
  scripts/validate_mac_mot_v2_checkpoint.py \
  --checkpoint <120k diffusion_pytorch_model.bin> \
  --model-config <120k config.json> \
  --device cuda:0 --smoke-forward
```

The scalar-head initialization and shared-prefix Q rejection review items
are implemented. Changing the former fully visible Q mask changes old Q
representations; existing online Q weights can warm-start further critic
training, but old Q scores cannot be assumed to retain their calibration.

Reproduce full-forward versus cached rejection timing and peak allocated GPU
memory with `scripts/benchmark_mac_prefix_cache.py --checkpoint <120k bin>
--model-config <120k config.json>`. It measures M=1/8/32, 512 language tokens,
288 image tokens, and ten Euler steps on synthetic observations. The reference
uses the corrected asymmetric mask but repeats the full prefix and computes
unused Value. Cache correctness is compared against that same-mask reference,
not against the old all-visible Q semantics.

Measured B200 results and numerical limitations are recorded in
[docs/MAC_PREFIX_CACHE_VALIDATION.md](docs/MAC_PREFIX_CACHE_VALIDATION.md).

Before declaring the complete MAC acceptance matrix finished, also run real
two-step distributed phase-1/phase-2 jobs, H=1 imagination, M=8/M=32 timing and
memory checks, critic checkpoint save/reload equality, and the same-seed
hanging-mug M=1 versus M=32 mini evaluation.

Data, caches, checkpoints, outputs, logs, credentials, and upstream source
trees are ignored and must never be committed.

## Legacy compatibility (`legacy_v1`)

Everything below describes the retained pre-MAC architecture. It is not the
current RL path.

### 120k pretraining lineage

The legacy 4B entrypoint is:

```text
robonana.configs.robotwin_flux2_4b_dino.config
```

It uses the official pretrained FLUX.2 Klein 4B backbone, hidden size 3072, 24
attention heads, 5 double-stream blocks, 20 single-stream blocks, BF16 ZeRO-2
training on eight GPUs, and a global batch of 256. It was trained for 120,000
steps on 2,500 Clean plus 25,000 Randomized FACT RoboTwin-v2 episodes.

The retained legacy token sequence is:

```text
[language | state | current_image | pred_action | gt_action_full_clean |
 idx_h | future_state | reward | success | Q |
 future_image_vae | future_image_dino]
```

Legacy training samples `idx_h` uniformly from `1..48`. The GT clean-action
track is causal and the horizon/world block can read only `G_1..G_idx_h`.
Reward is a direct scalar, success is BCE, and Q is a scalar flow token in the
shared FLUX stream. Future DINO is a final training-only auxiliary sink.

The legacy training command is:

```bash
export ROBONANA_GPU_IDS=0,1,2,3,4,5,6,7
export ROBONANA_BATCH_SIZE=16
export ROBONANA_MAX_STEPS=120000
export ROBONANA_PIXEL_EVAL_INTERVAL=2000
export ROBONANA_CHECKPOINT_INTERVAL=1000
export ROBONANA_BACKBONE_LR=2e-5
export ROBONANA_ROBOT_LR=1e-4

bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_4b_dino.config
```

DINOv3 targets are computed online from the three native RGB frames; DINO is
not cached and its frozen encoder is excluded from checkpoints and inference.

### Legacy inference modes

Only `legacy_v1` checkpoints support the old conditional world modes:

| mode | behavior |
|---|---|
| `action` | Stage-1 action diffusion only |
| `action_reward_q` | sample action, then legacy state/reward/success/Q queries |
| `world_all` | supplied action chunk, packed legacy horizons `1..48` |
| `world_horizon` | supplied action chunk and one legacy horizon |

`mac_mot_v2` live inference intentionally rejects
`action_reward_q`, `world_all`, and `world_horizon`; its learned world model is
used internally by phase-2 imagination.

The scratch 800M+DINO configuration remains only for tests, inexpensive smoke
runs, and ablations:

```bash
bash scripts/run_robotwin_train.sh \
  --config robonana.configs.robotwin_flux2_800m_dino.config
```

Removed full-model EMA, `td_posttrain`, `mc_posttrain`, and legacy scalar-Q
posttraining launch paths are intentionally unsupported.
