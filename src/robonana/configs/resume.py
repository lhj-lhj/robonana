"""显式续训入口；保留已验证的 FACT 恢复适配，不再在 import 时读取环境。"""
from dataclasses import dataclass
import json
from pathlib import Path
from .schema import accumulation


@dataclass(frozen=True)
class ResumeOptions:
    output: Path
    checkpoint: Path
    model_config: Path
    gpus: tuple[int, ...]
    microbatch: int
    accumulation_steps: int
    global_batch: int
    gradient_checkpointing: bool
    single_checkpoint_stride: int
    additional_steps: int
    max_steps: int
    universal_checkpoint: bool

    def __post_init__(self):
        if accumulation(self.gpus,self.microbatch,self.global_batch) != self.accumulation_steps:
            raise ValueError('gpus * microbatch * accumulation_steps must equal global_batch')
        if self.additional_steps < 0 or self.max_steps<=0: raise ValueError('Invalid resume budget')


def build_resume_config(o):
    from robonana.training.continuation import build_world_policy_resume, build_critic_continuation, restore_config_tuples
    source = restore_config_tuples(json.loads(o.model_config.read_text()))
    phase = source['train']['posttrain']['phase']
    if phase == 'world_policy':
        config = build_world_policy_resume(source, checkpoint=o.checkpoint, source_config=o.model_config,
            project_dir=o.output, gradient_checkpointing=o.gradient_checkpointing,
            single_checkpoint_stride=o.single_checkpoint_stride, additional_steps=o.additional_steps,
            gpu_ids=o.gpus, batch_size_per_gpu=o.microbatch, accumulation_steps=o.accumulation_steps,
            global_batch=o.global_batch)
    elif phase == 'critic':
        if o.additional_steps: raise ValueError('Critic continuation uses max_steps, not a restarted world LR curve')
        config = build_critic_continuation(source, checkpoint=o.checkpoint, source_config=o.model_config,
            project_dir=o.output, max_steps=o.max_steps, gpu_ids=o.gpus,
            batch_size_per_gpu=o.microbatch, accumulation_steps=o.accumulation_steps)
        config['models'].update(gradient_checkpointing=o.gradient_checkpointing,
                                gradient_checkpointing_single_stride=o.single_checkpoint_stride)
    else:
        raise ValueError(f'Unsupported saved phase: {phase}')
    if config['train']['max_steps'] != o.max_steps:
        raise ValueError('max_steps contradicts the saved budget + additional_steps')
    if o.universal_checkpoint:
        path=o.checkpoint/'deepspeed_universal.json'
        if not path.is_file(): raise FileNotFoundError(path)
        config['launch']['deepspeed_config']={'deepspeed_config_file':str(path)}
    return config
