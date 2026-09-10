#!/usr/bin/env python3
# 中文：隔离的 Stage1/预训练配对评测；结果不自动进入 replay。
# English: Paired evaluation orchestrator using existing collector and report.
# 调用 / Invocation: python script --help；创建独立评测目录 / creates isolated eval output.
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def evaluation_policies(args):
    """中文：显式选择运行范围。English: Never infer extra baseline runs."""
    policies = [('stage1', args.stage1, args.stage1_config)]
    if not args.stage1_only:
        policies.append(('pretrain', args.pretrain, args.pretrain_config))
    return policies


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--stage1-config',type=Path,required=True)
    p.add_argument('--pretrain',type=Path,required=True)
    p.add_argument('--pretrain-config',type=Path,required=True)
    p.add_argument('--seeds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robotwin',type=Path,required=True)
    p.add_argument('--sim-python',type=Path,required=True)
    p.add_argument('--initial-dataset',type=Path,required=True)
    p.add_argument('--server-gpu',type=int,default=4)
    p.add_argument('--sim-gpu',type=int,default=5)
    p.add_argument('--port',type=int,default=8394)
    p.add_argument('--stage1-only',action='store_true')
    p.add_argument('--adopt-running-fixed100',type=int,
                   help='Adopt this existing collector PID without restarting its episodes')
    args=p.parse_args()
    from robonana.inference_contract import read_contract
    contracts=[read_contract(ck)['sampling'] for ck in (args.stage1,args.pretrain)]
    for key in ('action_chunk','horizon','num_inference_steps','flow_shift'):
        if contracts[0][key]!=contracts[1][key]:
            raise ValueError(f'paired policy sampling mismatch: {key}')
    root=Path(__file__).resolve().parents[2]
    if args.adopt_running_fixed100 and not args.stage1_only:
        p.error('adopting a running collector requires --stage1-only')
    args.output.mkdir(parents=True,exist_ok=bool(args.adopt_running_fixed100))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=os.pathsep.join(str(root/x) for x in
        ('src','third_party/FACT','third_party/flux2/src','third_party/flux2_official/src'))
    def run(cmd,log,extra=None):
        with log.open('x') as f:
            subprocess.run(cmd,cwd=root,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT,check=True)
    # Prepare disjoint, expert-feasible held-out scenes. No policy-success filtering.
    held=args.output/'heldout_preflight'
    runtime=args.output/'seed_runtime'
    runtime.mkdir(mode=0o700,exist_ok=bool(args.adopt_running_fixed100))
    if not args.adopt_running_fixed100:
      run([str(args.sim_python),str(root/'scripts/internal/collect_robotwin_pool_worker.py'),
         '--prepare-seeds','20','--seed-start','300000','--robotwin',str(args.robotwin),
         '--output',str(held),'--port',str(args.port),'--worker-id','heldout',
         '--vector-env-checkout',str(root/'third_party/RoboTwin_RLinf')],args.output/'heldout_prepare.log',
         dict(CUDA_VISIBLE_DEVICES=str(args.sim_gpu),ROBONANA_SAPIEN_RENDER_DEVICE='cuda:0',
              OIDN_DEFAULT_DEVICE='cuda',ROBONANA_ROBOTWIN_STATIC_CAMERAS='head_camera',
              XDG_RUNTIME_DIR=str(runtime)))
    manifests={'fixed100':args.seeds,'heldout20':held/'accepted_seeds.json'}
    a=json.loads(args.seeds.read_text());b=json.loads(manifests['heldout20'].read_text())
    assert not ({j['seed'] for j in a['jobs']} & {j['seed'] for j in b['jobs']})
    for split,manifest in manifests.items():
        for name,ck,config in evaluation_policies(args):
            output=args.output/f'{split}_{name}'
            cmd=[sys.executable,str(root/'scripts/diagnostics/benchmark_robotwin_collection_pool.py'),
                 '--jobs-json',str(manifest),'--sim-gpus',str(args.sim_gpu),str(args.sim_gpu),
                 '--server-gpu',str(args.server_gpu),'--sim-python',str(args.sim_python),
                 '--robotwin',str(args.robotwin),'--checkpoint',str(ck),'--model-config',str(config),
                 '--initial-dataset',str(args.initial_dataset),'--output',str(output),
                 '--inference-mode','action_only','--inference-batch-size','2','--batch-wait-ms','10',
                 '--port',str(args.port),'--timeout-seconds','43200']
            if name=='stage1':cmd+=['--selected-world']
            if split=='fixed100' and args.adopt_running_fixed100:
                # 中文：只接管等待，不重启/终止正在采集的Stage1。
                # English: Wait for the orphaned collector, then validate its result.
                proc=Path(f'/proc/{args.adopt_running_fixed100}/cmdline')
                while proc.exists():
                    command=proc.read_bytes()
                    if not command:break  # exited zombie
                    if b'benchmark_robotwin_collection_pool.py' not in command or str(output).encode() not in command:
                        raise RuntimeError('adopted PID no longer identifies the expected collector')
                    time.sleep(15)
                result=json.loads((output/'summary.json').read_text())
                if len(result['episodes'])!=len(json.loads(manifest.read_text())['jobs']):
                    raise RuntimeError('adopted Stage1 evaluation did not complete all jobs')
            else:
                run(cmd,args.output/f'{split}_{name}.log')
            if name=='stage1':
                run([sys.executable,str(root/'scripts/report_selected_world_eval.py'),'--root',str(output/'selected_world')],
                    args.output/f'{split}_report.log')
        if args.stage1_only:continue
        results={name:json.loads((args.output/f'{split}_{name}'/'summary.json').read_text()) for name in ('stage1','pretrain')}
        byseed={name:{r['seed']:r for r in res['episodes']} for name,res in results.items()}
        assert byseed['stage1'].keys()==byseed['pretrain'].keys()
        rows=[]
        for seed,s in byseed['stage1'].items():
            base=byseed['pretrain'][seed]
            assert s['instruction']==base['instruction']
            rows.append(dict(seed=seed,instruction=s['instruction'],stage1=s['success'],pretrain=base['success']))
        (args.output/f'{split}_comparison.json').write_text(json.dumps(dict(
            stage1_sr=results['stage1']['success_rate'],pretrain_sr=results['pretrain']['success_rate'],episodes=rows),indent=2))
    (args.output/'complete.json').write_text(json.dumps({'splits':list(manifests),'paired':not args.stage1_only}))


if __name__=='__main__':main()
