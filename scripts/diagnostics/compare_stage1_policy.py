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
    args=p.parse_args()
    from robonana.inference_contract import read_contract
    contracts=[read_contract(ck)['sampling'] for ck in (args.stage1,args.pretrain)]
    for key in ('action_chunk','horizon','num_inference_steps','flow_shift'):
        if contracts[0][key]!=contracts[1][key]:
            raise ValueError(f'paired policy sampling mismatch: {key}')
    root=Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=os.pathsep.join(str(root/x) for x in
        ('src','third_party/FACT','third_party/flux2/src','third_party/flux2_official/src'))
    def run(cmd,log,extra=None):
        with log.open('x') as f:
            subprocess.run(cmd,cwd=root,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT,check=True)
    # Prepare disjoint, expert-feasible held-out scenes. No policy-success filtering.
    held=args.output/'heldout_preflight'
    runtime=args.output/'seed_runtime'
    runtime.mkdir(mode=0o700)
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
        for name,ck,config in [('stage1',args.stage1,args.stage1_config),('pretrain',args.pretrain,args.pretrain_config)]:
            output=args.output/f'{split}_{name}'
            cmd=[sys.executable,str(root/'scripts/diagnostics/benchmark_robotwin_collection_pool.py'),
                 '--jobs-json',str(manifest),'--sim-gpus',str(args.sim_gpu),str(args.sim_gpu),
                 '--server-gpu',str(args.server_gpu),'--sim-python',str(args.sim_python),
                 '--robotwin',str(args.robotwin),'--checkpoint',str(ck),'--model-config',str(config),
                 '--initial-dataset',str(args.initial_dataset),'--output',str(output),
                 '--inference-mode','action_only','--inference-batch-size','2','--batch-wait-ms','10',
                 '--port',str(args.port),'--timeout-seconds','43200']
            if name=='stage1':cmd+=['--selected-world']
            run(cmd,args.output/f'{split}_{name}.log')
            if name=='stage1':
                run([sys.executable,str(root/'scripts/report_selected_world_eval.py'),'--root',str(output/'selected_world')],
                    args.output/f'{split}_report.log')
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
    (args.output/'complete.json').write_text(json.dumps({'splits':list(manifests),'paired':True}))


if __name__=='__main__':main()
