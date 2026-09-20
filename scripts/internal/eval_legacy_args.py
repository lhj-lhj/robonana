#!/usr/bin/env python3
# 中文：旧 shell 参数适配，复用统一评测入口。
# English: Legacy arguments only, with no evaluation loop.
# 调用 / Invocation: Called by eval_robotwin_all_tasks_parallel.sh.
"""旧 shell 的参数适配器；不启动服务、不实现评测循环。"""
import os
from pathlib import Path
import sys


def arguments(config, episodes, env):
    def get(key, default):
        return env.get(key) or default
    checkpoint = Path(env['ROBONANA_TRAINED_CHECKPOINT'])
    model_config = get('ROBONANA_MODEL_CONFIG', str(next(
        (parent / 'config.json' for parent in checkpoint.parents if (parent / 'config.json').is_file()),
        checkpoint.parent.parent / 'config.json')))
    server = get('ROBONANA_EVAL_SERVER_GPUS', '0,1,2,3').split(',')
    sim = get('ROBONANA_EVAL_SIM_GPUS', '4,5,6,7').split(',')
    if len(server) != len(sim):
        raise ValueError('Policy and simulator GPU lists must have the same length')
    shared = server == sim
    if not shared and set(server) & set(sim):
        raise ValueError('Use identical colocated pools, or the pools must be disjoint')
    # 旧 seed group 仍映射到官方候选序列；不把基础设施失败算成 expert 拒绝。
    argv = ['eval', '--execute', '--checkpoint', str(checkpoint), '--model-config', model_config,
        '--output', get('ROBONANA_EVAL_RUN_DIR', 'outputs/robotwin_eval'),
        '--robotwin', get('ROBOTWIN_PATH', '/workspace/hongjia/RoboTwin'),
        '--initial-dataset', get('ROBONANA_DATASET_ROOT', '/workspace/datasets/fact-robotwin-v2/RoboTwin'),
        '--sim-python', get('ROBONANA_ROBOTWIN_PYTHON', get('ROBOTWIN_CONDA_ENV', '/data3/hongjia/venvs/robotwin-sapien303')+'/bin/python'),
        '--task-configs', config, '--episodes', str(episodes),
        '--seed-start', str(100000*(1+int(get('ROBONANA_EVAL_SEED_GROUP', '0')))),
        '--port', get('ROBONANA_PORT_BASE', '18000'),
        '--workers-per-gpu', get('ROBONANA_EVAL_JOBS_PER_GPU', '1'),
        '--seed-timeout', get('ROBONANA_EPISODE_TIMEOUT_SECONDS', '1200'),
        '--infra-retries', str(max(0, int(get('ROBONANA_EPISODE_GPU_ATTEMPTS', '3'))-1)),
        '--inference-mode', get('ROBONANA_INFERENCE_MODE', 'action_q_rejection'),
        '--capture-mode', 'full' if env.get('ROBONANA_ROLLOUT_DATASET_ROOT') else 'scout',
        '--gpus', *(server if shared else server+sim)]
    if shared:
        argv += ['--shared-gpus']
    if env.get('ROBONANA_EVAL_TASKS'):
        argv += ['--tasks', *env['ROBONANA_EVAL_TASKS'].split(',')]
    if env.get('ROBONANA_ROLLOUT_DATASET_ROOT'):
        argv += ['--export-dataset', env['ROBONANA_ROLLOUT_DATASET_ROOT']]
    return argv


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_multitask_mbrl import main
    sys.argv = [sys.argv[0], *arguments(sys.argv[1], sys.argv[2], os.environ)]
    main()
