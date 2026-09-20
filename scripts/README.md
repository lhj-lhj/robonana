# 启动与配置

只维护一个入口：`run_multitask_mbrl.py`。新训练、续训和评测都读显式 JSON，不接受旧 `ROBONANA_*` 实验环境变量；进程所需的 CUDA/NCCL、库路径、认证环境仍由运行环境管理。

## 先改配置，再看计划

| 配置 | 用途 | 必须核对 |
|---|---|---|
| [train.json](../configs/train.json) | 原始 FLUX 预训练 / Stage1 / Stage2 | 数据与模型路径、phase、步数、LR、batch、world_conditioning |
| [resume.json](../configs/resume.json) | 从完整保存点恢复同一阶段 | 来源 config/完整 checkpoint、总终点、追加步数、batch、是否 Universal |
| [eval.json](../configs/eval.json) | 成功率评测与失败数据采集 | 权重、仿真解释器、task/config、seed、episodes、并发和采集模式 |

示例中的 `assets/` 是需要你填写的资源位置，不会自动下载或猜服务器目录。相对路径以 **JSON 文件所在目录** 为准；仿真解释器保留 venv 符号链接。统计仍受现有 A 合同约束，文件里明确列出路径；换数据集不能自动换统计。

```bash
# 在已配置 FACT / RoboTwin 依赖的 Python 环境执行
python scripts/run_multitask_mbrl.py train --config configs/train.json
python scripts/run_multitask_mbrl.py eval --config configs/eval.json
python scripts/run_multitask_mbrl.py resume --config configs/resume.json
# 确认打印出的计划后，加 --execute 才真正启动
python scripts/run_multitask_mbrl.py train --config configs/train.json --execute
# 只核验训练数据/缓存，不启动训练
python scripts/run_multitask_mbrl.py audit --config configs/train.json
```

两个 shell 文件只调用当前 Python 并原样传递参数，不维护默认超参或兼容层。`train` 和 `resume` 创建独立输出，不覆盖来源实验。输出目录会保存 `requested.json` 与传给 FACT 的 `launch_config.json`；FACT 自己仍保存原生 `config.json`。评测保存完整 `eval_config.json`、冻结协议和每个 task/config 的 ledger。

## 不允许缺信息后静默猜测

- JSON **所有字段必须出现**，缺项、未知键、类型错误都会报错；`null` 只表示明确不用该可选输入，例如 pretrain 的来源 checkpoint，不会寻找历史保存点。
- `gpus`、`microbatch`、`accumulation_steps`、`global_batch` 都显式填写，检查 `GPU数 × microbatch × accumulation_steps = global_batch`。例如四卡32×累积2=256，八卡32×累积1=256。启动计划独立显示这四个值。
- `max_steps`、`warmup_steps`、`lr`、`robot_lr` 必须填写。scheduler 的 `decay_steps` 直接取同一个 `max_steps`；保存终点随预算联动，不再在多层 config 中覆盖。
- `world_conditioning` 同时写入模型和数据；动作仍固定48步。`sampling_steps`、`flow_shift`、`discount` 同时供训练、数据和保存的推理合同使用。
- 新阶段和续训不同：新 `train` 从0开始自己的 optimizer/LR 时钟；`resume` 明确继承来源文件的算法、LR与optimizer。world 续训 `additional_steps=0` 保留原预算，非零时沿用原峰值/模块追加一条曲线，且必须与填写的 `max_steps` 对得上。两者均打印完整最终配置。
- `smoke_steps` 是显式的1–10步短测开关，缩短预算时同步变更scheduler并在最终配置显示。常规训练填写0。

既有实验常用参数（供填写参考，**不是遗漏时的fallback**）：

| phase | max_steps | lr / robot_lr | 数据与初始化 |
|---|---:|---|---|
| pretrain | 120000 | 2e-5 / 1e-4 | 原始 FLUX；原成功示范 |
| stage1 | 60000 | 2e-5 / 2e-5 | 显式训练权重/config；原成功示范与最新失败各50% |
| stage2 | 20000 | 1e-4 / 1e-4 | 显式训练权重/config；冻结 FLUX，只训练 Q/Value |

Stage1/2必须同时提供 checkpoint、model_config、replay_root；Stage2不接受rope_prefix。`checkpoint_keeps` 是明确的保存里程碑，不能超出填写的训练终点。

## 评测只有一条执行链

`run_multitask_mbrl.py` → `internal/robotwin_eval_pool.py` → `internal/collect_robotwin_pool_worker.py`。共用模型服务、官方仿真和轨迹验收；诊断脚本也使用该组件。

- JSON 的 `tasks: []` 明确表示全部官方50任务；`task_configs` 明确选择clean/randomized；`episodes` 是每个 task/config 的数量。
- `seed_start=100000`：官方 seed=0 的候选起点。expert 明确不可解才跳候选；超时/未知错误重试原seed，耗尽后留 `blocked.json`、其余任务继续。
- `inference_mode` 明确选择 `action_only` 或 `action_q_rejection`，不从命令名猜策略；采样语义从已认证checkpoint合同读取并校验。
- `capture_mode=scout` 只测SR；`scout_replay` 只保存动作精确回放验收通过的失败；`full` 保存全部。`export_dataset` 可将已验收full轨迹导出给 `prepare_robotwin_rollouts.py`，无需另一套采集入口。
- `shared_gpus=true` 配合 `workers_per_gpu: [2]`：每卡一份batch=1模型、两个独立仿真进程，公共队列动态领取 task/config；同一配置内seed保持顺序。
- 固定 expert manifest 跨主机分片仍使用该管线，要求显式 shard_count，暂限每卡1 worker。
- 输出目录有独占锁。旧进程已退出后，`resume_interrupted=true` 归档半成品再试原seed；不删除既有结果。旧协议不兼容时明确报错，不自动改账本。

## 保留的工具

- `prepare_robotwin_rollouts.py`：现有回放索引与缓存，不重新拟合统计。
- `report_selected_world_eval.py`、`diagnostics/probe_mac_world_fit.py`：图像/World离线诊断。
- `diagnostics/compare_stage1_policy.py`：同seed模型比较，复用内部评测组件。
- `diagnostics/prepare_universal_checkpoint.py`：复用DeepSpeed官方转换生成独立恢复视图。
- `diagnostics/benchmark_*`、`validate_*`：有界性能/正确性测试；不是另一套正式训练协议。
- `services/`：现有推理协议服务；`env/`：FACT/RoboTwin/SAPIEN环境适配。必要的上游环境变量仅在进程边界组装，不作为用户实验配置。

旧单任务轮次、legacy参数翻译、孤立eval启动器和7个叠加配置模块已删除；源码历史由Git保留。历史实验文档中的旧命令不可作为新启动方式。
