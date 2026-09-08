# 脚本导航 / Script guide

顶层只保留 6 个常用正式入口。子目录是专业工具或内部辅助，不是多套算法。
Only six common public entry points remain at the top level. Subdirectories
contain specialist tools and helpers, not alternative algorithms.

从仓库根目录运行命令，使用 190 已配置好的 Python 环境。Python 工具可用
`--help` 查看参数（`env/robotwin_eval_bootstrap.py` 是例外，它转发执行参数）。
Run commands from the repository root with the configured Python environment
on 190. Python tools expose `--help`, except the environment bootstrap, which
forwards execution arguments. Shell launchers are operational commands, not
read-only help commands: inspect their headers and the project README first.

## 常用入口 / Public entry points

| 文件 / File | 用途 / Purpose | 副作用 / Effects |
|---|---|---|
| [run_robotwin_train.sh](run_robotwin_train.sh) | 单阶段训练 / Single-phase training | 启动训练，写日志和权重 / Trains; writes logs and checkpoints |
| [run_hanging_mug_mac_round.sh](run_hanging_mug_mac_round.sh) | 完整 MAC 轮次 / Full MAC round | 两阶段训练、评测和采集 / Trains, evaluates and collects |
| [collect_prepare_robotwin_rollouts.sh](collect_prepare_robotwin_rollouts.sh) | 采集并准备回放 / Collect and prepare replay | 仿真、轨迹和缓存 / Simulation, rollouts and caches |
| [eval_robotwin_all_tasks_parallel.sh](eval_robotwin_all_tasks_parallel.sh) | 并行成功率评测 / Parallel success-rate evaluation | 推理服务、仿真和评测结果 / Services, simulation and eval outputs |
| [prepare_robotwin_rollouts.py](prepare_robotwin_rollouts.py) | 处理已有回放 / Prepare existing replay | 索引与缓存，不训练 / Index/cache writes, no training |
| [report_selected_world_eval.py](report_selected_world_eval.py) | 生成重建报告 / Generate reconstruction report | 从已有产物写 HTML / Writes HTML from existing artifacts |

单次训练用 `run_robotwin_train.sh`，不要直接找内部的 `train_robotwin.py`。
阶段与超参数由配置决定，采样和输入参数受 checkpoint 契约约束。
Use `run_robotwin_train.sh`, not the internal Python launcher, for ordinary
training. Config selects the phase; checkpoint contracts constrain sampling
and model inputs. See the [project README](../README.md) for training commands.

## 子目录 / Tool categories

### data — 数据准备 / Data preparation

- `convert_120k_action_checkpoint.py`: 显式一次性导出归档 actor，保留 action segment，限制 action-only；不认证旧训练输入，也不恢复 legacy 运行时。 / One-time archived actor export, preserving action segments; action-only execution, not historical training certification.

- `preprocess_robotwin_lerobot_flux.py`: 原始 LeRobot 的语言/图像缓存 / Original LeRobot language/image caches.
- `preprocess_robotwin_flux.py`: HDF5 语言/图像缓存，回放入口复用它 / HDF5 caches, also reused by replay preparation.
- `compute_robotwin_lerobot_metadata.py`: 原始数据索引与统计来源维护 / Original-data indexing and statistics provenance maintenance.

两种源格式仍使用同一图像预处理与 A 统计。不要为回放重拟合统计；不要把元数据工具当作日常训练前置步骤。
Both source formats use the same image pipeline and A statistics. Do not refit
statistics for replay or treat metadata maintenance as a daily training step.

### diagnostics — 按需诊断 / Opt-in diagnostics

- `probe_mac_world_fit.py`: 训练集固定窗口重建 / Fixed-window training-set reconstruction.
- `start_mac_world_pilot.py`: 有界拟合实验，**确实会启动训练** / Bounded fitting pilot; **does start training**.
- `benchmark_mac_prefix_cache.py`: Q 前缀缓存数值与速度 / Q-prefix parity and timing.
- `benchmark_mac_world_cache.py`: World model 前缀缓存数值与速度 / World-prefix parity and timing.
- `benchmark_robotwin_inference_batch.py`: 推理 batch 数值与速度 / Inference-batch parity and timing.
- `benchmark_robotwin_collection_pool.py`: 并行采集吞吐，会写测试轨迹 / Collection throughput; writes probe rollouts.
- `verify_image_pipeline.py`: 真实 VAE 缓存/在线一致性 / Real-VAE cache/live parity.
- `validate_mac_mot_v2_checkpoint.py`: 权重结构与可选 forward / Weight structure and optional forward check.
- `validate_mac_distributed_safety.py`: 小模型多卡保护/恢复测试 / Tiny-model distributed guards and resume tests.
- `validate_robotwin_lerobot_flux.py`: 原始元数据和缓存检查 / Original metadata and cache validation.
- `audit_robotwin_instructions.py`: 语言指令审查，也是评测预检 / Instruction audit, also an eval preflight.
- `stress_sapien_oidn.py`: 仿真渲染压力测试 / Simulator-rendering stress test.

诊断结果不自动进入训练数据池；拟合测试不等于泛化成功率。
Diagnostic outputs are not automatically added to replay; fitting probes are
not generalization success-rate evaluations.

### env — 环境适配与维护 / Environment adaptation and maintenance

- `robotwin_eval_python.sh`: 选择仿真解释器 / Select the simulator interpreter.
- `robotwin_eval_bootstrap.py`: SAPIEN 与 RoboTwin 启动适配 / SAPIEN/RoboTwin bootstrap.
- `install_sapien_oidn_blackwell.sh`: 显式安装修复，**会修改环境** / Explicit repair installation; **modifies the environment**.
- `verify_remote.sh`: 指定 `PYTHON_BIN` 后运行测试 / Run tests using `PYTHON_BIN`.

### services — 协议服务 / Protocol services

- `inference_server_robotwin_batched.py`: 正式评测/采集使用的动态 batch 服务 / Dynamic batching used by evaluation and collection.
- `inference_server_robotwin.py`: 单请求 FACT TCP 集成 / Single-request FACT TCP integration.
- `inference_server_robotwin_xpolicylab.py`: XPolicyLab 协议集成 / XPolicyLab protocol integration.

三者复用同一 policy 和 checkpoint 契约，不维护独立算法或采样默认值。
All three reuse the same policy and checkpoint contract, without independent
algorithms or sampling defaults.

### internal — 内部辅助 / Internal helpers

- `train_robotwin.py`: Shell 训练入口调用的 FACT launcher 适配 / FACT launcher adapter called by the training shell entry.
- `eval_robotwin_task_isolated.py`: 并行评测调用的单任务隔离执行器 / Isolated task runner called by parallel evaluation.
- `collect_robotwin_pool_worker.py`: 并行采集管理器启动的 worker / Worker spawned by the collection supervisor.

## 路径迁移 / Path migration

保留上述 6 个顶层入口名称；其他脚本已直接移动，不保留旧路径包装器。
仓库内当前调用与测试已更新。历史实验文档可能记录旧路径，按本表定位同名脚本；
历史运行目录、权重和数据未改动。外部手写命令需要换成新路径。
The six public names are preserved; other scripts moved without compatibility
wrappers. Current callers and tests are updated. Dated experiment documents
may show old paths: locate the same filename in the categories above. Existing
run directories, weights and data are unchanged; update external manual commands.
