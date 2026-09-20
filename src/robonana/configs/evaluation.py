"""评测的唯一配置表。所有字段必须显式提供，不保留运行时默认值。"""
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
    gpus: tuple[int, ...]
    shared_gpus: bool
    workers_per_gpu: tuple[int, ...]
    tasks: tuple[str, ...]
    task_configs: tuple[str, ...]
    episodes: int
    seed_start: int
    seed_timeout: int
    candidate_multiplier: int
    infra_retries: int
    port: int
    inference_mode: str
    capture_mode: str
    candidate_batch_size: int
    manifests: Path | None
    expert_seed_cache: Path | None
    shard_count: int | None
    shard_offset: int
    ready_only: bool
    allow_partial_expert_seeds: bool
    resume_interrupted: bool
    export_dataset: Path | None

    def __post_init__(self):
        if self.inference_mode not in ('action_only','action_q_rejection'): raise ValueError('Invalid inference_mode')
        if self.capture_mode not in ('scout','scout_replay','full'): raise ValueError('Invalid capture_mode')
        if not self.task_configs or set(self.task_configs)-{'demo_clean','demo_randomized'}:
            raise ValueError('task_configs must select clean/randomized')
        if not 1<=self.candidate_batch_size<=32: raise ValueError('candidate_batch_size must be 1..32')
        if not 1<=self.port<=65535-len(self.gpus)+1: raise ValueError('Invalid policy port range')
        if self.expert_seed_cache and self.shard_count is None:
            raise ValueError('Cached-seed evaluation requires explicit shard_count')
