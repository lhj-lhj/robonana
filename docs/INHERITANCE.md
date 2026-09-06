# RoboNana inheritance and model boundary

RoboNana keeps one official FLUX.2 backbone and adds robot-specific adapters,
world heads, masks, and deterministic critic experts. It does not vendor or
fork FACT, FLUX.2, MAC, or ImageWAM source.

## Upstream reuse

| Upstream | Reused directly | RoboNana-owned extension |
|---|---|---|
| FACT | collator, trainer lifecycle, Accelerate/DeepSpeed integration | RoboTwin loaders, losses, eval and checkpoint hooks |
| FLUX.2 | text/image projections, RoPE, modulation, double/single blocks, image final layer | action/state projections, token layout and attention masks |
| MAC | one-rollout targets, online-Q action selection, Value-only target update | fixed-48 RoboTwin reward and world-model wiring |
| ImageWAM | MoT expert shape, `prepare_qkv -> mixed attention -> apply_post`, slim initialization policy | one-query deterministic scalar Value/Q experts |

Pinned ImageWAM reference:

- https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a
- https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/action_dit_flux2.py
- https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/src/imagewam/models/backbones/mot.py#L612-L745
- https://github.com/yuyangalin/ImageWAM/blob/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a/scripts/flux2/preprocess_action_dit_flux2.py

Original MAC reference:

- https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L191-L217
- https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L262-L318

The implementation files repeat these links next to the adapted logic.

## Maintained `mac_mot_v2` model

The actor/world FLUX sequence is exactly:

```text
[language | state | current_image_vae | pred_action | clean_action_chunk |
 reward | success | future_state | future_image_vae]
```

It has a fixed action horizon of 48 and no `idx_h`, Value, Q, or DINO token.
With `C=[language,state,current_image]`, `A=pred_action`, and
`G=clean_action_chunk`, the shared-FLUX dependencies are:

| Query | Readable keys |
|---|---|
| `C` | `C` |
| `A` | `C,A` |
| `G` | `C,G` |
| reward | `C,G,reward` |
| success | `C,G,reward,success` |
| future state | `C,G,reward,success,future state` |
| future image | `C,G,reward,success,future state,future image` |

The predicted-action track is never readable by the clean world path. The
world cascade is therefore
`reward -> success -> future_state -> future_image_vae`.

## Deterministic MoT critics

Value and Q are independent slim experts; each owns exactly one learned query
and produces one scalar. They do not receive noise, timesteps, action-flow
tokens, reward/success tokens, or future targets.

During critic training the complete actor/world FLUX is frozen and runs under
`torch.no_grad()`. At every double- and single-stream layer, RoboNana caches
the frozen FLUX K/V before the backbone residual update. The matching expert
layer prepares its query Q/K/V, concatenates frozen prefix K/V with its own K/V,
computes attention only for the expert query, and applies the expert residual
path. This is the ImageWAM MoT information-flow pattern specialized to a
deterministic scalar.

- Value FLUX prefix: `[language,state,current_image]`.
- Q FLUX prefix: `[language,state,current_image,clean_action_chunk]`.
- The condition C reads only C; a complete clean-action G reads C and G.
  Q queries read C, their own G, and their own query. Candidates cannot read
  other candidates. This is the actor/world mask specialized to the critic.
- No expert output or gradient is fed back into FLUX.
- No Q target or Q EMA exists.
- The only target network is an FP32 copy of `value_expert`.

The 4B default expert width is 1024. Its attention retains the main FLUX head
count and per-head width; MLP/residual width is slimmed to 1024. Expert
initialization follows ImageWAM's preprocessing policy: exact tensor copy when
shapes match, axis-wise linear interpolation otherwise, and fan-in alpha
scaling when the final input width changes. Only the expert body is transferred.
Each learned query and the entire scalar head (output linear layer and head
AdaLN modulation) retain their fresh initialization. ImageWAM similarly skips
its task-specific `action_encoder.*` and `head.*`; RoboNana's learned query
replaces that input encoder. FLUX `final_layer.*` is never mapped to a scalar
head. This applies only to initial migration: exact loading of trained MAC
checkpoints preserves all online expert parameters across rounds.

### Shared condition K/V execution

`prefill_condition_cache` computes C once per observation at clean timestep 0.
The actor reads this same cache across all Euler steps and candidate groups;
only the noisy action stream receives a changing action timestep. Q scoring
re-encodes the final clean chunk with the clean segment/position convention.

The action branch reuses the official FLUX `_prepare_qkv`/`_apply_residuals`
and `_qkv`/`_out` methods. C K/V stay at observation batch size B; candidate
branches store only their own 48-token K/V and a reference to C. Combined K/V
are materialized one layer at a time for a bounded candidate group. Attention
uses one softmax over all visible C/action/query keys.

`sample_q_rejection` uses Q-only scoring and defaults to groups of eight
(`ROBONANA_REJECTION_CANDIDATE_BATCH_SIZE`). Joint critic-loss forwards share C
between online V and Q; imagined next-state online and target V share another
C cache. FLUX cache construction is no-grad, while online experts remain in
the DDP forward/autograd graph. Caches are request-local and never reused after
an observation or FLUX weight change.

This changes Q's former fully visible prefix semantics. Existing trained Q
weights can warm-start training with the corrected mask; old Q scores are not
guaranteed to stay calibrated. Parameter names/shapes and exact loading remain
unchanged. Equivalence tests use the full forward with the corrected mask.
See [MAC_PREFIX_CACHE_VALIDATION.md](MAC_PREFIX_CACHE_VALIDATION.md) for tests,
real 4B smoke results, performance measurements, and BF16 numerical differences.

## Two serial optimization phases

`world_policy` trains FLUX plus actor/world adapters on real data. The action
loss is multiplied by the recorded success mask; failures still train all
world losses.

`critic` freezes every non-expert parameter. It samples online-policy action
candidates, uses online Q to select one, performs exactly one learned-world
transition, and minimizes deterministic MSE for online V and online Q. The
targets are stop-gradient:

```text
V target = R_chunk + gamma^48 * nonterminal * target_value(next_state)
Q target = R_chunk + gamma^48 * nonterminal * online_value(next_state)
```

Only a successful optimizer step updates target Value by Polyak averaging.

## Checkpoint boundary

The first `mac_mot_v2` run migrates from the immutable 120k
`legacy_v1` checkpoint. The migration whitelist includes the official FLUX
backbone and shape-compatible image/action/state adapters. It explicitly skips
old horizon, segment, Q-flow, Value-token, DINO, and other project heads, even
when a shape happens to match. Both new experts are then initialized from the
loaded FLUX weights.

Subsequent `mac_mot_v2` rounds use exact trained loading with their saved
`config.json`. The target Value expert is stored separately from the online
model. A full-model EMA file and target-Q file are invalid for this
architecture.

## Retained legacy boundary

`legacy_v1` remains only for first-stage variable-`idx_h` pretraining,
strict loading, and historical inference. Its token order and losses are not
silently reinterpreted as `mac_mot_v2`. The older full-model-EMA,
`td_posttrain`, `mc_posttrain`, and Q-flow RL paths have been removed.
