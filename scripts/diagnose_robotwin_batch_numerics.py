#!/usr/bin/env python3
"""Read-only checkpoint diagnosis: encoders, fixed-feature isolation and flow.

Calls existing policy/sampler and the independent full-forward benchmark oracle.
Backend toggles below are local controlled experiments, never production config.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from PIL import Image
import torch

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from robonana.inference.robotwin_policy import seeded_randn_like
from robonana.sampling import sample_q_rejection
from world_action_model import apply_runtime_compat
from world_action_model.pipeline.utils import normalize_state
from benchmark_mac_prefix_cache import full_rejection


def error(a, b):
    return float((a.float() - b.float()).abs().max())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Replay all original rows with only VAE features substituted')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.probe_config.read_text())
    apply_runtime_compat()
    policy = BatchedRoboNanaRobotWinPolicy(**{k: cfg[k] for k in (
        'checkpoint', 'model_config', 'flux_checkpoint_dir', 'stats_path')},
        model_device='cuda:0', vae_device='cuda:0', text_encoder_device='cuda:0',
        dtype=torch.float32, num_inference_steps=20, rejection_candidate_count=32)
    observations = []
    for item in cfg['observations']:
        with h5py.File(item['source'], 'r') as f:
            obs = {'instruction': str(f.attrs['instruction']),
                   'observation.state': f['joint_action/vector'][item['frame']],
                   'sampling_seed': item['sampling_seed']}
            for target, source in [('cam_high','head_camera'), ('cam_left_wrist','left_camera'),
                                   ('cam_right_wrist','right_camera')]:
                with Image.open(BytesIO(bytes(f[f'observation/{source}/rgb'][item['frame']]))) as im:
                    obs[f'observation.images.{target}'] = np.array(im.convert('RGB'))
            observations.append(obs)
    reports = []
    def report(name, **values):
        row = dict(name=name, **values)
        reports.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    report('runtime', commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           torch=torch.__version__, cuda=torch.version.cuda,
           cudnn_tf32=torch.backends.cudnn.allow_tf32,
           matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
           matmul_precision=torch.get_float32_matmul_precision(),
           vae_dtype=str(next(policy.vae.parameters()).dtype))
    # Same observation order as the original four-observation numerical probe.
    solo = torch.cat([policy._batched_current_image_tokens([o]) for o in observations])
    pair = torch.cat([policy._batched_current_image_tokens(observations[k:k+2]) for k in (0,2)])
    report('vae_default', max_abs=error(solo,pair), per_row=[error(a,b) for a,b in zip(solo,pair)])
    old = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        solo_ieee = torch.cat([policy._batched_current_image_tokens([o]) for o in observations])
        pair_ieee = torch.cat([policy._batched_current_image_tokens(observations[k:k+2]) for k in (0,2)])
        report('vae_no_tf32', max_abs=error(solo_ieee,pair_ieee),
               solo_change=error(solo,solo_ieee), batch_change=error(pair,pair_ieee))
    finally:
        torch.backends.cudnn.allow_tf32 = old
    contexts = [policy._batched_context([o])[0] for o in observations]
    # Cold language batching is independent from FLUX masking; keep encoder dtype.
    embedder = policy._text_embedder
    texts = [observations[0]['instruction'], observations[2]['instruction'] + ' carefully near the rack.']
    lang_single = torch.cat([embedder([text]) for text in texts])
    lang_batch = embedder(texts)
    report('qwen_cold_batch', max_abs=error(lang_single,lang_batch), shape=list(lang_batch.shape))
    state = normalize_state(torch.tensor(np.stack([o['observation.state'] for o in observations]),
        device='cuda',dtype=torch.float32), policy.normalization, mode='zscore')[:,None]
    noise = torch.stack([torch.stack([seeded_randn_like(torch.zeros(1,48,14,device='cuda'),
        int(o['sampling_seed'])+1009*k)[0] for k in range(32)]) for o in observations])
    model = policy.model
    if args.aggregate_only:
        def replay(images, batch, group):
            outputs, qs, best = [], [], []
            for start in range(0,len(observations),batch):
                ids=list(range(start,min(start+batch,len(observations))))
                c=torch.cat([contexts[k] for k in ids])
                result=sample_q_rejection(model=model,context=c,current_latents=images[ids],state=state[ids],
                    context_mask=torch.ones(c.shape[:2],device='cuda',dtype=torch.bool),candidate_count=32,
                    action_noise=noise[ids],schedule=policy.schedule,grid_height=12,grid_width=24,
                    candidate_batch_size=group)
                outputs.append(result.candidates);qs.append(result.candidate_q);best.append(result.best_index)
            return torch.cat(outputs),torch.cat(qs),torch.cat(best)
        baseline=replay(solo,1,16)
        for name,images in [('aggregate_fixed_features',solo),('aggregate_live_vae',pair)]:
            result=replay(images,2,32)
            report(name,action_error=error(baseline[0],result[0]),
                   action_errors_per_row=[error(a,b) for a,b in zip(baseline[0],result[0])],
                   q_error_return_units=error(baseline[1],result[1])*1000,
                   indices=result[2].tolist(),reference_indices=baseline[2].tolist())
        return
    reference = {}
    original = model.predict_action_cached
    original_prefill = model.prefill_condition_cache

    def run(name, ids, images=solo, pad=0, mutate_partner=False):
        c = torch.cat([contexts[k] for k in ids])
        im, st, n = images[ids].clone(), state[ids].clone(), noise[ids].clone()
        mask = torch.ones(c.shape[:2],device='cuda',dtype=torch.bool)
        if pad:
            c = torch.cat([c, torch.full((len(ids),pad,c.shape[-1]),123.,device='cuda')],dim=1)
            mask = torch.cat([mask,torch.zeros(len(ids),pad,device='cuda',dtype=torch.bool)],dim=1)
        arow = ids.index(0)
        if mutate_partner:
            brow = 1-arow
            c[brow] = 17.; im[brow] = -3.; st[brow] = 2.; n[brow] = .5
        traces, caches = [], []
        def trace(cache, action, **kw):
            out = original(cache, action, **kw)
            traces.append(out[arow*32:(arow+1)*32].cpu())
            return out
        def prefill(**kw):
            out = original_prefill(**kw); caches.append(out); return out
        model.predict_action_cached, model.prefill_condition_cache = trace, prefill
        try:
            result = sample_q_rejection(model=model,context=c,current_latents=im,state=st,
                context_mask=mask,candidate_count=32,action_noise=n,schedule=policy.schedule,
                grid_height=12,grid_width=24,candidate_batch_size=32)
        finally:
            model.predict_action_cached, model.prefill_condition_cache = original, original_prefill
        if not reference:
            reference.update(result=result, traces=traces, cache=caches[0])
        cache_errors=[]
        if not pad:
            for stream in ('double','single'):
                for layer, (a,b) in enumerate(zip(getattr(reference['cache'],stream),getattr(caches[0],stream))):
                    cache_errors.append(max(error(a[k][0],b[k][arow]) for k in ('k','v')))
        report(name, candidate_action_error=error(reference['result'].candidates[0],result.candidates[arow]),
               q_error_return_units=error(reference['result'].candidate_q[0],result.candidate_q[arow])*1000,
               selected=int(result.best_index[arow]), prefix_kv_errors=cache_errors,
               velocity_errors=[error(a,b) for a,b in zip(reference['traces'],traces)])
        return result
    run('fixed_single_A',[0])
    run('fixed_repeat_A',[0])
    run('fixed_AB',[0,2])
    run('fixed_BA',[2,0])
    run('fixed_AB_mutated_B',[0,2],mutate_partner=True)
    run('fixed_AB_masked_padding',[0,2],pad=8)
    run('live_vae_AB',[0,2],images=pair)
    # Compare cache path against existing full asymmetric-forward oracle on two
    # independent samples (one candidate each keeps the reference memory bounded).
    inputs=dict(context=torch.cat([contexts[0],contexts[2]]),current_latents=solo[[0,2]],
                state=state[[0,2]],context_mask=torch.ones(2,contexts[0].shape[1],device='cuda',dtype=torch.bool))
    n=noise[[0,2],:1]
    full=full_rejection(model,inputs,n,policy.schedule)
    cached=sample_q_rejection(model=model,**inputs,action_noise=n,candidate_count=1,
        candidate_batch_size=1,schedule=policy.schedule,grid_height=12,grid_width=24)
    report('full_vs_cache',action_error=error(full.candidates,cached.candidates),
           q_error_return_units=error(full.candidate_q,cached.candidate_q)*1000)


if __name__=='__main__':
    main()
