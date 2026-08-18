# Inheritance and reuse map

## Runtime graph

```text
FACT LeRobotDataset
  -> FACT WATransformsLerobot
    -> RoboNanaTransforms (adds horizon_idx only)
      -> Flux2FACTModel (subclasses official Flux2)
        -> existing Flux2.double_blocks
        -> existing Flux2.single_blocks
        -> existing Flux2.final_layer for image output
        -> small action/state/value heads
```

## Reused without copying

| Upstream | Reused code | robonana extension |
|---|---|---|
| FACT | `fact_datasets` and RoboTwin sampling | none |
| FACT | `WATransformsLerobot` normalization/failure fields | `RoboNanaTransforms` adds `horizon_idx` |
| FACT | `CasualWATrainer` orchestration contract | later thin trainer override |
| FLUX.2 | `Flux2`, image/text projections, RoPE and modulation | `Flux2FACTModel` subclass |
| FLUX.2 | all `DoubleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | all `SingleStreamBlock` parameters | masked forward using existing private helpers |
| FLUX.2 | image `final_layer` | action/state/value linear heads |

## New trainable parameters

- shared action input projection for noisy and clean action tracks;
- shared state input projection for current and future state;
- value input projection;
- horizon embedding and segment embeddings;
- action, future-state, and value output heads.

There is no MoT, ActionDiT, or second transformer backbone.

## Token order

```text
[language | state | current image | noisy action | clean GT action |
 horizon | future state | value | future image]
```

The attention mask is applied inside every reused FLUX.2 double-stream and single-stream block.

## Offline cache path

```text
RoboTwin instruction -> official FLUX.2 Qwen3Embedder.forward -> language_context.pt
RoboTwin HDF5 cameras -> FACT build_robotwin_three_view_tensor -> FLUX.2 AE -> frame tokens
```

The cache adds no alternative preprocessing geometry. It calls FACT's existing
three-view layout helper and reproduces the official FLUX.2 Klein VAE
patchify/BatchNorm sequence. One episode tensor is indexed as both
`current_latent[t]` and `future_latent[min(t + idx_h, T - 1)]`.
