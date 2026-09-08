#!/usr/bin/env python3
# 中文：发布固定训练场景清单；只合并已通过expert预检的seed/指令，不合并轨迹。
# English: Publish a fixed scene manifest from expert-validated seeds/instructions, not trajectories.
# 调用 / Invocation: 输入预检JSON，输出新清单；拒绝重复seed和覆盖已有清单。 / Reads preflight JSONs; refuses duplicate seeds or overwrites.
"""Lock an ordered seed/instruction fixture and its simulator configuration."""
import argparse
import json
from pathlib import Path
import subprocess

from robonana.inference_contract import sha256_file
from robonana.sim.collection_pool import validate_jobs


def merge_manifests(payloads, expected_count):
    signatures = {(p['task_name'], p['task_config']) for p in payloads}
    if len(signatures) != 1 or any(p.get('expert_validated') is not True for p in payloads):
        raise ValueError('Only one expert-validated task/config is permitted')
    jobs = sorted((job for p in payloads for job in p['jobs']), key=lambda job: int(job['seed']))
    validate_jobs(jobs, 1)
    if len(jobs) != expected_count:
        raise ValueError(f'Expected {expected_count} unique scenes, got {len(jobs)}')
    task, config = signatures.pop()
    return dict(task_name=task, task_config=config, jobs=jobs, expert_validated=True,
                scene_count=len(jobs), purpose='fixed_training_scenes_not_heldout_eval', version=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=100)
    parser.add_argument('--robotwin', type=Path, required=True)
    args = parser.parse_args()
    result = merge_manifests([json.loads(p.read_text()) for p in args.inputs], args.expected_count)
    result['source_manifest_sha256'] = {str(p.resolve()): sha256_file(p) for p in args.inputs}
    result['robotwin_checkout'] = str(args.robotwin.resolve())
    result['robotwin_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.robotwin, text=True).strip()
    result['task_config_sha256'] = sha256_file(args.robotwin / 'task_config' / (result['task_config'] + '.yml'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(dict(path=str(args.output), scenes=result['scene_count'], sha256=sha256_file(args.output))))


if __name__ == '__main__':
    main()
