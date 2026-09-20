# 脚本导航

当前实验命令、模型路径和两组消融见 [实验文档](../docs/MULTITASK_MBRL_PROTOCOL.md)。从仓库根目录运行，190使用已配置好的Python环境。Python入口可先看 `--help`；Shell入口通常会直接启动任务。

## 常用入口

| 文件 | 做什么 | 会不会占GPU/写文件 |
|---|---|---|
| [run_multitask_mbrl.py](run_multitask_mbrl.py) | 50任务训练、采集、评测；`--world-conditioning fixed48` 或 `rope_prefix` 选择两组world训练 | `train` 默认只打印计划，`--execute` 才启动；`audit` 读取实际数据 |
| [eval_robotwin_all_tasks_parallel.sh](eval_robotwin_all_tasks_parallel.sh) | 兼容旧环境变量，转发到统一 Python 评测入口 | 会启动推理和仿真 |
| [run_robotwin_train.sh](run_robotwin_train.sh) | 按配置训练一个阶段 | 会训练并保存权重 |
| [run_hanging_mug_mac_round.sh](run_hanging_mug_mac_round.sh) | 历史单任务MAC轮次 | 会依次训练、评测、采集；不是当前50任务入口 |
| [collect_prepare_robotwin_rollouts.sh](collect_prepare_robotwin_rollouts.sh) | 采集并准备回放 | 会仿真、写轨迹、生成缓存 |
| [prepare_robotwin_rollouts.py](prepare_robotwin_rollouts.py) | 处理已有回放 | 写索引和缓存，不训练 |
| [report_selected_world_eval.py](report_selected_world_eval.py) | 从已有结果生成重建报告 | 写HTML |

单任务入口的 `ROBONANA_MAC_TRAIN_ONLY=1` 仍会自动跑完Stage1后进入Stage2。新project设置 `ROBONANA_RESUME=0` 时只加载来源权重，optimizer、scheduler和step从头开始。预算分别用 `ROBONANA_MAC_WORLD_POLICY_STEPS` / `ROBONANA_MAC_CRITIC_STEPS`，默认保留各阶段最近两个checkpoint。

## 唯一评测管线

`run_multitask_mbrl.py` → `internal/robotwin_eval_pool.py` → `internal/collect_robotwin_pool_worker.py`。旧 shell 和诊断命令只做参数转发。

- `collect` 默认 action-only + `scout_replay`（只保存通过精确动作回放验收的失败）；`eval` 历史默认 action-Q + `scout`。可显式用 `--inference-mode action_only --capture-mode scout_replay`，避免依赖命令名猜策略。
- `--shared-gpus --gpus 0 1 2 3 4 5 6 7 --workers-per-gpu 2`：八份模型、最多16个独立仿真进程，推理仍 batch=1。也可逐卡设置数量；默认1，最多4。
- 公共队列按 task/config 领取，空闲 worker 继续领取下一配置。配置内部仍按官方 seed 顺序检查；默认候选100000，只有 expert 明确不可解才跳过。随机数、物理步数、48步动作、OIDN和失败回放方式保持一致。
- `--infra-retries 2`：超时/异常原 seed 最多重试两次，单独保留日志；耗尽后该配置写 `blocked.json`，其他配置继续执行，最终退出码非零；不伪造失败率或换 seed。
- `protocol.json` 冻结模型/seed/仿真版本；`execution.json` 记录并发。输出有独占运行锁。旧运行停止后，`--resume-interrupted` 可将未提交 attempt 移到 `interrupted/`，再重试原 seed。旧 ledger 原样保留，历史被超时跳过的 seed 不会自动补回。
- 已收集 expert manifest、跨机器分片仍复用同一管线；固定分片暂限每卡1 worker，防止重复 seed。
- 不引入异步视频/PNG编码、另一个模型服务实现或常驻多场景仿真框架；先用小规模实测决定并发数。

## 按需工具

### data：数据和权重转换

- `convert_120k_action_checkpoint.py`：显式导出旧actor，保留action segment，只支持action-only；不证明旧训练输入与当前一致。
- `publish_robotwin_scene_manifest.py`：固定seed和指令清单，绑定仿真版本及配置哈希。
- `preprocess_robotwin_lerobot_flux.py` / `preprocess_robotwin_flux.py`：原始LeRobot / HDF5回放的图像与语言缓存。
- `compute_robotwin_lerobot_metadata.py`：原始数据索引及统计来源维护。

两种数据源共用图像管线和A统计。回放不重新拟合统计。

### diagnostics：诊断与短测

- `benchmark_robotwin_collection_pool.py`：采集吞吐测试；支持 `--jobs-json`、`--inference-mode action_only`、`--capture-mode scout`。会启动仿真并写结果。
- `compare_stage1_policy.py`：固定场景的Stage1/120k action-only对照及重建。
- `probe_mac_world_fit.py`：固定训练窗口重建；`start_mac_world_pilot.py` 会启动有界训练。
- `benchmark_mac_prefix_cache.py` / `benchmark_mac_world_cache.py`：前缀缓存的数值和速度。
- `benchmark_robotwin_inference_batch.py`：推理batch的数值和速度。
- `verify_image_pipeline.py`：真实VAE缓存/在线一致性；`stress_sapien_oidn.py`：渲染压力测试。
- `validate_mac_mot_v2_checkpoint.py`：权重结构和可选forward；`validate_mac_distributed_safety.py`：小模型多卡保护与恢复。
- `validate_robotwin_lerobot_flux.py` / `audit_robotwin_instructions.py`：元数据、缓存、语言指令检查。
- `prepare_universal_checkpoint.py`：DeepSpeed重分片准备，保留来源checkpoint。
- `train_action_student.py`：独立一步action实验，`--launch` 会训练并评测；见 [历史方案](../docs/archive/STAGE1_STUDENT_SPLIT_20260910.md)。

Stage2计时用 `benchmark_mac_world_cache.py --stage2-breakdown --batch-size 1 --repeats 5`，需完整权重、模型/数据配置；`--target-value` 指定EMA。它会更新临时模型但不保存。共享GPU短测排除加载和I/O，不能直接外推成八卡正式训练速度。

### env / services / internal

- `env/robotwin_eval_python.sh` 选择仿真解释器；`robotwin_eval_bootstrap.py` 适配SAPIEN/RoboTwin。
- `env/install_sapien_oidn_blackwell.sh` 会修改环境；`verify_remote.sh` 按 `PYTHON_BIN` 跑测试。
- `services/inference_server_robotwin_batched.py` 是正式动态batch服务；另有单请求FACT TCP、XPolicyLab协议适配，三者共用policy。
- `internal/robotwin_eval_pool.py` 管理仿真进程与轨迹验收；`collect_robotwin_pool_worker.py` 执行官方 expert check 和仿真。旧 `eval_robotwin_task_isolated.py` 只保留参数转发和日志/进程辅助，不再维护评测循环。

历史文档可能使用迁移前的脚本路径，按本表找同名文件。诊断输出不自动进入训练池，训练集拟合也不代表环境成功率。
