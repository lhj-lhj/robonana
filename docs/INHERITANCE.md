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
- No expert output or gradient is fed back into FLUX.
- No Q target or Q EMA exists.
- The only target network is an FP32 copy of `value_expert`.

The 4B default expert width is 1024. Its attention retains the main FLUX head
count and per-head width; MLP/residual width is slimmed to 1024. Expert
initialization follows ImageWAM's preprocessing policy: exact tensor copy when
shapes match, axis-wise linear interpolation otherwise, and fan-in alpha
scaling when the final input width changes. The learned query remains new.

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
