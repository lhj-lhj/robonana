#!/usr/bin/env python3
# 中文：一次性迁移旧 120k actor；不修改原权重，不恢复旧运行时。
# English: One-time legacy actor export; preserve source weights and current runtime.
# 调用 / Invocation: 显式指定输入和新输出目录；写转换权重与审计记录。 / Explicit inputs and a new output directory; writes weights and audit metadata.
"""Export the archived bidirectional actor into fixed-48 MAC for action-only use.

The old actor attends only to L/S/I/A, never horizon, value or DINO tokens.
Preserving the first four segment rows is essential; the former training
warm-start converter initialized them afresh and is NOT an actor-preserving export.
New world classifiers/critics are untrained. Runtime metadata describes current
execution, NOT historical training-input certification. No legacy loader is added.
"""
import argparse
import json
from pathlib import Path

import torch
from flux2.model import Flux2Params

from robonana.inference_contract import build_contract, sha256_file, write_contract
from robonana.models.mac_flux2_fact import MacFlux2FACTModel


def migrate_actor(source, model):
    """Fail closed on unknown/missing backbone tensors; initialize new roots only."""
    obsolete = {"value_in", "value_out", "horizon_embed", "segment_embed",
                "dino_in", "dino_out", "dino_segment_embed"}
    new_roots = {"reward_token", "success_token", "reward_out", "success_out",
                 "actor_world_segment_embed", "value_expert", "q_expert"}
    target = model.state_dict()
    copied, skipped = {}, []
    for name, tensor in source.items():
        if name.split('.')[0] in obsolete:
            skipped.append(name)
            continue
        if name not in target or tensor.shape != target[name].shape:
            raise ValueError(f"Unrecognized/shape-mismatched source tensor: {name}")
        copied[name] = tensor
    missing = set(target) - set(copied)
    if any(name.split('.')[0] not in new_roots for name in missing):
        raise ValueError(f"Missing actor/backbone tensors: {sorted(missing)}")
    old_segments = source.get('segment_embed.weight')
    if old_segments is None or old_segments.shape != target['actor_world_segment_embed.weight'].shape:
        raise ValueError('Expected the original eight-row segment embedding')
    model.load_state_dict(copied, strict=False, assign=True)
    for root in sorted(new_roots):
        module = getattr(model, root)
        module.to_empty(device='cpu')
        module.reset_parameters()
    with torch.no_grad():
        model.actor_world_segment_embed.weight[:4].copy_(old_segments[:4])
        # Retain the shared state/image segment semantics as an initialization;
        # this does not certify the changed world-model information flow.
        model.actor_world_segment_embed.weight[6].copy_(old_segments[5])
        model.actor_world_segment_embed.weight[7].copy_(old_segments[7])
    return dict(copied=sorted(copied), skipped=sorted(skipped),
                initialized=sorted(missing), segment_rows_copied={'0': 0, '1': 1, '2': 2, '3': 3, '6': 5, '7': 7})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--reference-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vae-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.source_config.read_text())
    config = json.loads(args.reference_config.read_text())
    source_model = old['models']
    if not source_model.get('pred_action_bidirectional') or source_model['max_horizon'] != 48:
        raise ValueError('Only archived bidirectional fixed-length 48 actor is supported')
    for key in ('num_inference_steps', 'flow_shift'):
        target_key = 'sampling_steps' if key == 'num_inference_steps' else key
        if old['train'][key] != config['train']['posttrain']['imagination'][target_key]:
            raise ValueError(f'Archived actor sampling differs from requested runtime: {key}')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    with torch.device('meta'):
        model = MacFlux2FACTModel(Flux2Params(**source_model['params']),
                                 action_dim=source_model['action_dim'], state_dim=source_model['state_dim'])
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    report = migrate_actor(source, model)
    models = dict(source_model, architecture_version='mac_mot_v2', chunk_horizon=48,
                  reward_dim=48, success_dim=1, q_dim=1, value_dim=1,
                  reward_head_type='binary_chunk', dino_dim=None, expert_hidden_dim=1024)
    config['models'] = models
    # Export metadata is not a runnable training configuration.
    (args.output / 'model_config.json').write_text(json.dumps({'models': models}, indent=2))
    weights = args.output / 'diffusion_pytorch_model.bin'
    torch.save(model.state_dict(), weights)
    report.update(source=str(args.source.resolve()), source_sha256=sha256_file(args.source),
                  historical_training_inputs_certified=False,
                  note='Actor-preserving weight map; current preprocessing/FP32 may differ from historical inference.')
    (args.output / 'conversion_report.json').write_text(json.dumps(report, indent=2))
    contract = build_contract(config['train']['posttrain'], args.vae_checkpoint)
    contract['capabilities'] = ['action_only']
    contract['historical_training_inputs_certified'] = False
    contract['conversion_source_sha256'] = report['source_sha256']
    write_contract(weights, contract, phase='converted_action_only', step=120000)
    print(json.dumps({'weights': str(weights), 'copied': len(report['copied']),
                      'capabilities': contract['capabilities']}), flush=True)


if __name__ == '__main__':
    main()
