#!/usr/bin/env python3
# 中文：独立蒸馏实验入口；不修改/保存教师，不接入 Stage 2。
# English: Isolated two-GPU student experiment, not a new MAC training phase.
"""Distill final actions from the frozen production 20-step teacher.

MAC agents/mac.py::bc_actor_loss supplies the same-noise supervised objective:
https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L70-L101
Reuse RoboNana datasets, four-pool sampler, FLUX cache, Euler sampling and MoT
blocks. No GT-action loss, world loss, Q guidance or trainable teacher here.
Student FP32 master weights + BF16 autocast; FP32 MSE and Adam state.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import ConcatDataset, DataLoader

from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset, RoboTwinPosttrainSampler
from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
from robonana.models.flux2_action_student import build_action_student
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import prefill_mac_condition, sample_action_flow, flow_euler_schedule
from robonana.training.continuation import restore_config_tuples
from robonana.inference_contract import sha256_file


def heldout_episode(record):
    # 中文：按整个 episode 留出，禁止相邻滑窗跨 train/validation 泄漏。
    # English: Stable episode split; validation noise uses a separate generator.
    identity = f"{record.source.resolve()}:{record.episode_index}"
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 10 == 0


def student_pe(teacher, batch, device):
    ids = teacher._robot_ids(batch_size=batch, length=48, segment_id=3,
        device=device, dtype=torch.long,
        time_ids=torch.arange(1, 49, device=device)[None].expand(batch, -1))
    return teacher.pe_embedder(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--hidden-dim', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        p.error('positive step/batch/intervals required')
    acc = Accelerator(mixed_precision='bf16', log_with=None if args.offline else 'wandb')
    set_seed(args.seed)
    if acc.is_main_process:
        args.output.mkdir(parents=True, exist_ok=False)
    acc.wait_for_everyone()
    cfg = restore_config_tuples(json.loads(args.config.read_text()))
    teacher, _ = load_flux2_fact_trained_checkpoint(args.teacher, config_path=args.config,
                                                   device=acc.device, dtype=torch.bfloat16)
    teacher.eval().requires_grad_(False)
    student = build_action_student(teacher, args.hidden_dim).to(acc.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(.9, .95),
                                 weight_decay=1e-4, fused=True)
    warmup = min(100, max(1, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / warmup, 1.) * .5 * (1 + math.cos(math.pi * max(0, step-warmup) / max(1,args.steps-warmup))))
    classes = {c.__name__: c for c in (RoboTwinHDF5Dataset, RoboTwinLeRobotDataset)}
    children, validation, manifest = [], [], []
    for pool in cfg['dataloaders']['train']['data_or_config']:
        child = classes[pool['_class_name']].load(pool)
        child.open()
        records = list(child.records)
        val = [i for i, rec in enumerate(records) if heldout_episode(rec)]
        # Cache held-out observations only; GT actions never become labels.
        for i in val[:3]:
            index = int((child.episode_starts[i] + child.episode_stops[i] - 1) // 2)
            validation.append(child[index])
        manifest += [dict(source=str(rec.source), episode=rec.episode_index,
                          heldout=heldout_episode(rec), pool=pool['pool_name']) for rec in records]
        child._set_records([rec for rec in records if not heldout_episode(rec)])
        children.append(child)
    if not validation:
        raise ValueError('No held-out episodes; refuse to report training states as validation')
    dataset = ConcatDataset(children)
    sampler_cfg = {k:v for k,v in cfg['dataloaders']['train']['sampler'].items()
                   if k not in ('type','_class_name','batch_size','dataset')}
    sampler_cfg.update(seed=args.seed, infinite=False,
                       sample_epoch_size=args.steps * args.batch_size * acc.num_processes)
    sampler = RoboTwinPosttrainSampler(dataset, batch_size=args.batch_size * acc.num_processes, **sampler_cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    student, optimizer, loader = acc.prepare(student, optimizer, loader)
    # Keep scheduler local: one step per optimizer update, never multiply by ranks.
    metadata = {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'teacher_frozen':True,'student_only':True,'world_size':acc.num_processes,
                'global_batch':args.batch_size * acc.num_processes,
                'teacher_sampling':cfg['train']['posttrain']['imagination'],
                'precision':'FP32 student master/Adam; BF16 autocast/teacher; FP32 loss'}
    if acc.is_main_process:
        metadata['teacher_sha256'] = sha256_file(args.teacher)
        (args.output/'config.json').write_text(json.dumps(metadata,indent=2))
        (args.output/'split.json').write_text(json.dumps(manifest,indent=2))
    if not args.offline:
        acc.init_trackers('robonana', config=metadata,
                          init_kwargs={'wandb':{'name':args.output.name}})
    imag = cfg['train']['posttrain']['imagination']
    schedule = flow_euler_schedule(imag['sampling_steps'],flow_shift=imag['flow_shift'],device=acc.device)
    def inputs(item):
        return dict(context=item['context'].to(acc.device,dtype=torch.bfloat16),
            current_latents=item['current_latents'].to(acc.device,dtype=torch.bfloat16),
            state=item['state'].to(acc.device,dtype=torch.bfloat16)[:,None],
            context_mask=item['context_mask'].to(acc.device),
            grid_height=cfg['train']['latent_grid_height'],grid_width=cfg['train']['latent_grid_width'])
    def target_and_cache(item, generator):
        condition=inputs(item)
        with torch.no_grad(), acc.autocast():
            cache=prefill_mac_condition(model=teacher,**condition)
            n=condition['state'].shape[0]
            noise=torch.randn(n,48,teacher.action_dim,device=acc.device,dtype=torch.bfloat16,generator=generator)
            target=sample_action_flow(action_noise=noise, schedule=schedule,
                predict_action=lambda action,sigma:teacher.predict_action_cached(cache,action,
                    batch_indices=torch.arange(n,device=acc.device),timestep=sigma))
            pe=student_pe(teacher,n,acc.device)
        return cache,noise,target,pe
    @torch.no_grad()
    def validate(step):
        student.eval()
        values=[]
        gen=torch.Generator(device=acc.device).manual_seed(args.seed+900000)
        for item in validation[:8]:
            repeated={k:v[None].repeat_interleave(8,0) for k,v in item.items() if torch.is_tensor(v)}
            cache,noise,target,pe=target_and_cache(repeated,gen)
            with acc.autocast(): pred=student(cache,noise=noise,query_pe=pe)
            mse=(pred.float()-target.float()).square().mean()
            td=target.float().var(0,unbiased=False).mean().sqrt()
            sd=pred.float().var(0,unbiased=False).mean().sqrt()
            values.append(torch.stack([mse,td,sd]))
        means=acc.reduce(torch.stack(values).mean(0),reduction='mean')
        metrics=dict(validation_mse=means[0].item(),teacher_diversity_rms=means[1].item(),
            student_diversity_rms=means[2].item(),diversity_ratio=(means[2]/means[1].clamp_min(1e-8)).item())
        acc.log(metrics,step=step)
        if acc.is_main_process:
            with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**metrics))+'\n')
            print(json.dumps(dict(step=step,**metrics)),flush=True)
        student.train()
    validate(0)
    gen=torch.Generator(device=acc.device).manual_seed(args.seed+acc.process_index)
    tic=time.perf_counter()
    for step,item in enumerate(loader,1):
        cache,noise,target,pe=target_and_cache(item,gen)
        with acc.autocast(): prediction=student(cache,noise=noise,query_pe=pe)
        loss=(prediction.float()-target.float()).square().mean()
        bad=acc.reduce((~torch.isfinite(loss)).to(torch.int32),reduction='sum')
        if bad.item():raise FloatingPointError('non-finite distillation loss on a rank')
        acc.backward(loss)
        norm=acc.clip_grad_norm_(student.parameters(),1.)
        bad=acc.reduce((~torch.isfinite(norm)).to(torch.int32),reduction='sum')
        if bad.item():raise FloatingPointError('non-finite student gradient')
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        if any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError('teacher gradient leakage')
        if step%10==0:
            metrics=dict(distill_loss=acc.reduce(loss.detach(),reduction='mean').item(),
                         lr=scheduler.get_last_lr()[0],seconds_per_step=(time.perf_counter()-tic)/10)
            acc.log(metrics,step=step)
            if acc.is_main_process:print(json.dumps(dict(step=step,**metrics)),flush=True)
            tic=time.perf_counter()
        del cache,noise,target,pe,prediction,loss
        if step%args.eval_every==0 or step==args.steps:validate(step)
        if step%args.save_every==0 or step==args.steps:
            acc.save_state(str(args.output/f'step_{step:06d}'))
        if step>=args.steps:break
    acc.wait_for_everyone()
    if acc.is_main_process:
        (args.output/'complete.json').write_text(json.dumps({'step':step,'teacher_unchanged':True}))
    for child in children:child.close()
    acc.end_training()


if __name__=='__main__':
    main()
