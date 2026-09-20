"""评测的唯一配置表。JSON 必须完整填写；默认值仅用于生成可见示例。"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalOptions:
    output: Path
    checkpoint: Path
    model_config: Path
    robotwin: Path
    sim_python: Path
    initial_dataset: Path
    flux_checkpoint_dir: Path
    stats_path: Path
    gpus: tuple[int, ...] = tuple(range(8))
    shared_gpus: bool = True
    workers_per_gpu: tuple[int, ...] = (2,)
    tasks: tuple[str, ...] = ()
    task_configs: tuple[str, ...] = ('demo_clean','demo_randomized')
    episodes: int = 50
    seed_start: int = 100000
    seed_timeout: int = 1200
    candidate_multiplier: int = 20
    infra_retries: int = 2
    port: int = 9900
    inference_mode: str = 'action_only'
    capture_mode: str = 'scout_replay'
    candidate_batch_size: int = 32
    manifests: Path | None = None
    expert_seed_cache: Path | None = None
    shard_count: int | None = None
    shard_offset: int = 0
    ready_only: bool = False
    allow_partial_expert_seeds: bool = False
    resume_interrupted: bool = False
    export_dataset: Path | None = None

    def __post_init__(self):
        if self.inference_mode not in ('action_only','action_q_rejection'): raise ValueError('Invalid inference_mode')
        if self.capture_mode not in ('scout','scout_replay','full'): raise ValueError('Invalid capture_mode')
        if not self.task_configs or set(self.task_configs)-{'demo_clean','demo_randomized'}:
            raise ValueError('task_configs must select clean/randomized')
        if not 1<=self.candidate_batch_size<=32: raise ValueError('candidate_batch_size must be 1..32')
        if not 1<=self.port<=65535-len(self.gpus)+1: raise ValueError('Invalid policy port range')
        if self.expert_seed_cache and self.shard_count is None:
            raise ValueError('Cached-seed evaluation requires explicit shard_count')
