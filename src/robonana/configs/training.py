"""唯一新训练配置：扁平参数 → 一次组装 FACT 字典。无环境变量、无 import 时配置求值。"""
from dataclasses import dataclass
from pathlib import Path
import sys
from .schema import accumulation

# 这些是模型/动作合同，不是每个实验各写一份的超参。
CHUNK = 48
ACTION_DIM = 14
MODEL_PARAMS = dict(in_channels=128, context_in_dim=7680, hidden_size=3072, num_heads=24,
                    depth=5, depth_single_blocks=20, axes_dim=[32,32,32,32], theta=2000,
                    mlp_ratio=3.0, use_guidance_embed=False)
PHASES = ('pretrain', 'stage1', 'stage2')
LOSS_FIELDS = frozenset(("image_loss", "action_loss", "future_state_loss", "reward_loss",
                         "success_loss", "value_loss", "q_loss"))


@dataclass(frozen=True)
class TrainOptions:
    # 路径必须显式提供；不再猜 hongjia 的数据目录或某次历史实验。
    output: Path
    dataset_root: Path
    flux_checkpoint_dir: Path
    stats_path: Path
    phase: str
    architecture_version: str
    expert_hidden_dim: int
    checkpoint: Path | None
    model_config: Path | None
    replay_root: Path | None
    task_globs: tuple[str, ...]
    replay_task_glob: str
    collection_round: int
    # 四个量全部可见且必须相互一致；缺信息不能静默猜 batch 或累积次数。
    gpus: tuple[int, ...]
    microbatch: int
    global_batch: int
    accumulation_steps: int
    num_workers: int
    max_steps: int
    warmup_steps: int
    lr: float
    robot_lr: float
    world_conditioning: str
    gradient_checkpointing: bool
    single_checkpoint_stride: int
    checkpoint_interval: int
    checkpoint_total_limit: int
    checkpoint_keeps: tuple[int, ...]
    seed: int
    log_interval: int
    log_with: str | None
    tracker_project: str
    tracker_entity: str | None
    run_name: str | None
    discount: float
    sampling_steps: int
    flow_shift: float
    train_candidates: int
    eval_candidates: int
    loss_weights: dict[str, float]
    smoke_steps: int
    smoke_save: bool
    expected_original_episodes: int | None
    expected_tasks: int | None

    def __post_init__(self):
        if accumulation(self.gpus, self.microbatch, self.global_batch) != self.accumulation_steps:
            raise ValueError("gpus * microbatch * accumulation_steps must equal global_batch")
        if self.max_steps is None or self.lr is None or self.robot_lr is None:
            raise ValueError("max_steps, lr and robot_lr must be explicit")
        if self.architecture_version not in ('mac_mot_v2', 'mac_mot_v3'):
            raise ValueError('architecture_version must be mac_mot_v2 or mac_mot_v3')
        if self.expert_hidden_dim <= 0:
            raise ValueError('expert_hidden_dim must be positive')
        if self.architecture_version == 'mac_mot_v3' and self.world_conditioning != 'fixed48':
            raise ValueError('v3 currently requires fixed48')
        if self.phase not in PHASES: raise ValueError('phase must be pretrain/stage1/stage2')
        if self.world_conditioning not in ('fixed48','rope_prefix'): raise ValueError('Invalid world_conditioning')
        if self.phase == 'stage2' and self.world_conditioning != 'fixed48': raise ValueError('Stage2 requires fixed48')
        if self.phase != 'pretrain' and not all((self.checkpoint,self.model_config,self.replay_root)):
            raise ValueError('Stage1/Stage2 require explicit source checkpoint, model_config and replay_root')
        if self.phase == 'pretrain' and any((self.checkpoint,self.model_config,self.replay_root)):
            raise ValueError('Pretrain starts from original FLUX; use stage1 or resume for trained weights')
        if not self.task_globs: raise ValueError('task_globs cannot be empty')
        if self.collection_round < 0 or self.num_workers < 0: raise ValueError('round/workers must be nonnegative')
        if self.smoke_steps < 0 or self.smoke_steps > 10 or self.smoke_save and not self.smoke_steps:
            raise ValueError('Smoke saving requires a bounded smoke budget of 1..10')
        if min(self.single_checkpoint_stride,self.checkpoint_interval,self.checkpoint_total_limit,self.log_interval) < 1:
            raise ValueError('Checkpoint stride/interval/retention and log interval must be positive')
        if self.max_steps is not None and self.max_steps <= 0: raise ValueError('max_steps must be positive')
        if self.warmup_steps < 0 or not self.smoke_steps and self.warmup_steps >= self.steps:
            raise ValueError('warmup_steps must be below max_steps')
        if set(self.loss_weights) != LOSS_FIELDS or any(v < 0 for v in self.loss_weights.values()):
            raise ValueError('loss_weights must specify the seven maintained nonnegative weights')
        if not self.smoke_steps and any(s < 1 or s > self.steps for s in self.checkpoint_keeps):
            raise ValueError('checkpoint_keeps must be inside the training budget')
        if any(v is not None and v <= 0 for v in (self.lr,self.robot_lr)): raise ValueError('Learning rates must be positive')

    @property
    def steps(self):
        return self.smoke_steps or self.max_steps


def build_training_config(o: TrainOptions):
    """只在这里映射到 FACT：修改一个参数会同步影响所有消费者。"""
    import copy
    phase = 'critic' if o.phase == 'stage2' else 'world_policy' # o.phase取值是pretrain、stage1或stage2
    #  组装数据池（Data Mixture），如果是 stage1 / stage2：额外挂载 HDF5 失败回放池（成功与失败各 50% 混合）
    weights = dict(original_success=1. if o.phase=='pretrain' else .5,
                   collected_success_replay=0., historical_failure_replay=0.,
                   latest_failure=0. if o.phase=='pretrain' else .5)
    # o.world_conditioning = “fixed48”或“rope_prefix“
    shared = dict(stats_path=str(o.stats_path), action_chunk=CHUNK, action_dim=ACTION_DIM,
        max_horizon=CHUNK, fixed_horizon=CHUNK, discount=o.discount, reward_non_goal=-1.,
        reward_goal=0., q_target_mode='mac_mot_v2', world_conditioning=o.world_conditioning)

    original = dict(shared, _class_name='RoboTwinLeRobotDataset', data_path=str(o.dataset_root),
        index_path=str(o.dataset_root/'robonana_index.json'), task_globs=o.task_globs,
        episode_filter='success', pool_name='original_success', allow_empty=False, require_final_observation=False)
    pools = [original]

    if o.phase != 'pretrain':
        # 数据源字段单复数差异只在这个适配边界处理，实验配置不用跟着数据类换名字。
        # 加入replay的成功数据，历史失败数据和最近的失败数据。
        for name, filtering in [('collected_success_replay','success'),('historical_failure_replay','failure'),('latest_failure','failure')]:
            pool = dict(shared, _class_name='RoboTwinHDF5Dataset', data_path=str(o.replay_root),
                index_path=str(o.replay_root/'robonana_index.json'), task_glob=o.replay_task_glob,
                pool_name=name, episode_filter=filtering, allow_empty=name!='latest_failure', require_final_observation=True)
            if name=='historical_failure_replay': pool['round_max']=o.collection_round-1
            if name=='latest_failure': pool['round_id']=o.collection_round
            pools.append(pool)

    # 根据不同阶段构造优化器      
    sampler = (dict(type='RoboTwinEpisodeSampler', infinite=True) if o.phase=='pretrain' else
        dict(type='RoboTwinPosttrainSampler', infinite=True, pool_weights=dict(weights),
             redistribute_empty_historical_failure_to_latest=True, redistribute_empty_collected_success_to_original=True))

    posttrain = dict(enabled=True, algorithm='mac_mot_v2', q_target_mode='mac_mot_v2', phase=phase,
        chunk_horizon=CHUNK, discount=o.discount, reward_non_goal=-1., reward_goal=0., return_scale=1000.,
        current_collection_round=o.collection_round,
        ema=dict(decay=.995, update_every_optimizer_steps=1, start_step=0, target='value_expert_only'),
        imagination=dict(rollout_chunks=1, candidate_count=o.train_candidates, sampling_steps=o.sampling_steps,
            flow_shift=o.flow_shift, candidate_selection='argmax_q', fresh_each_batch=True, stop_gradient_target=True),
        data_mixture=dict(weights, success_only_action_bc=True, all_real_rollouts_train_world=True,
            redistribute_empty_historical_failure_to_latest=True, redistribute_empty_collected_success_to_original=True),
        environment_policy=dict(candidate_count=o.eval_candidates, candidate_selection='argmax_q',
            action_chunk=CHUNK, execute_actions_per_plan=CHUNK))
    
    from robonana.inference_contract import sampling_contract
    sampling_contract(posttrain)

    keeps = [] if o.smoke_steps else sorted(set((*o.checkpoint_keeps, o.steps)))
    # WandB 看板名
    tracker = dict(name=o.run_name or f'multitask-mbrl-loop{o.collection_round}-{o.phase}-{o.world_conditioning}')
    if o.tracker_entity: tracker['entity']=o.tracker_entity

    #  .../RoboNANA/runtime
    repo = Path(__file__).resolve().parents[3]
    return dict(
        project_dir=str(o.output), runners=['robonana.training.robotwin_trainer.RoboNanaTrainer'],
        launch=dict(gpu_ids=list(o.gpus), distributed_type='DEEPSPEED',
            deepspeed_config=dict(deepspeed_config_file=str(repo/'third_party/FACT/fact_train/distributed/accelerate_configs/zero2.json')),
            executable=f'{sys.executable} -m accelerate.commands.accelerate_cli', until_completion=False),
        dataloaders=dict(train=dict(data_or_config=original if o.phase=='pretrain' else pools,
            batch_size_per_gpu=o.microbatch, num_workers=o.num_workers, pin_memory=True,
            persistent_workers=o.num_workers>0, prefetch_factor=4 if o.num_workers>0 else None,
            transform=None, sampler=sampler, collator=dict(is_equal=True)), test={}),
        models=dict(architecture_version=o.architecture_version, initialization='flux_backbone' if o.phase=='pretrain' else 'trained',
            checkpoint=str(o.flux_checkpoint_dir/'flux-2-klein-base-4b.safetensors') if o.phase=='pretrain' else str(o.checkpoint),
            checkpoint_config=None if o.phase=='pretrain' else str(o.model_config), checkpoint_dir=str(o.flux_checkpoint_dir),
            params=copy.deepcopy(MODEL_PARAMS), action_dim=ACTION_DIM, state_dim=ACTION_DIM, reward_dim=CHUNK,
            success_dim=1, q_dim=1, reward_head_type='binary_chunk', max_horizon=CHUNK,
            pred_action_bidirectional=True, chunk_horizon=CHUNK, value_dim=1, dino_dim=None,
            expert_hidden_dim=o.expert_hidden_dim, train_mode=phase, world_conditioning=o.world_conditioning,
            # 复用模型现有部分重计算开关；不改变精度或动作/World注意力规则。
            gradient_checkpointing=o.gradient_checkpointing, gradient_checkpointing_single_stride=o.single_checkpoint_stride,
            vae_dtype='float32'),
        optimizers=dict(type='AdamW', lr=o.lr,
            robot_lr=o.robot_lr,
            betas=(.9,.95), eps=1e-8, weight_decay=1e-4, fused=True, foreach=False),
        # 预算仅定义一次，checkpoint终点和调度器衰减随 steps 联动。
        schedulers=dict(type='WarmupCosineScheduler', warmup_steps=1 if o.smoke_steps else o.warmup_steps, decay_steps=o.steps),
        train=dict(max_steps=o.steps, gradient_accumulation_steps=accumulation(o.gpus,o.microbatch,o.global_batch),
            mixed_precision='bf16', activation_checkpointing=False, checkpoint_interval=1 if o.smoke_save else o.checkpoint_interval,
            early_checkpoint_steps=(), checkpoint_keeps=keeps, checkpoint_total_limit=o.checkpoint_total_limit,
            checkpoint_save_optimizer=True, disable_checkpointing=bool(o.smoke_steps and not o.smoke_save),
            resume=False, allow_uncertified_pretrain=False, seed=o.seed, log_with=o.log_with,
            tracker_project_name=o.tracker_project, tracker_init_kwargs=dict(wandb=tracker),
            log_interval=1 if o.smoke_steps else o.log_interval, latent_grid_height=12, latent_grid_width=24,
            flow_shift=o.flow_shift, num_inference_steps=o.sampling_steps, max_grad_norm=1., memory_limit_gib=0.,
            discount=o.discount, reward_non_goal=-1., reward_goal=0., q_target_mode='mac_mot_v2',
            loss_weights=dict(o.loss_weights), with_ema=False, posttrain=posttrain))
