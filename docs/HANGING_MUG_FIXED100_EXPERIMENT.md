# Hanging mug：固定 100 个训练场景的 MAC 实验

更新：2026-09-09。执行主机：190；代码：`/data3/hongjia/robonana`。
本地：`D:\Robotic\robonana`。代码经 GitHub main 同步，权重/数据不提交 Git。

## 1. 固定实验协议 / Protocol

```text
固定 100 个训练场景 seed + 对应 instruction
  → 120k 预训练 actor，action_only，收集 round 0：100 条真实轨迹
  → 记录初始成功率 SR_0
  → 原始 Clean 示范 + 截至当前轮的累积真实回放
  → Stage 1：20,000 optimizer steps
      成功轨迹：action BC + world-model loss
      失败轨迹：仅 world-model loss
  → Stage 2：10,000 optimizer steps
      冻结整个 FLUX；训练 online Q / online Value；仅 Value 有 EMA target
  → 当前 actor 采样候选，argmax Q 选动作
  → 在相同 100 个 seed + instruction 上重新收集 100 条真实轨迹
  → 独立保存本轮数据、逐 seed 结果、成功率和 checkpoint
  → 新真实回放加入累积数据池，进入下一轮
```

`round r` 表示采集轮次。Round 0 没有新 RL 训练，由原始 120k actor 初始化。
消费 round 0 数据的两阶段更新产生下一次采集策略。20k/10k 是每个新阶段的
新增 optimizer steps，不是终身 checkpoint 步数。当前仅执行 round 0，未启动训练。

`SR_r = 本轮成功 episode 数 / 100`。这是固定**训练场景**上的成功率，不是独立
未见场景泛化指标。每个 seed 每轮恰好执行一次，失败也保存；不重试到成功。
基础设施异常与真实任务失败分开记录，未完成的轨迹不能算完整 episode。

## 2. 场景、输入和数据

| 项目 | 本实验约定 |
|---|---|
| task / 环境 | `hanging_mug` / `demo_clean` |
| 场景数 | 100 个唯一、官方 expert 可行性检查通过的训练 seed |
| instruction | 每个 seed 一条固定 seen instruction；后续不重新抽选 |
| 原始示范 | `/workspace/datasets/fact-robotwin-v2/RoboTwin`，仅 `Clean/hanging_mug` 的50条；不混入此前500条Randomized |
| 累积回放 | 本实验各轮完整成功/失败轨迹；不自动混入旧实验或测速轨迹 |
| 归一化 | A：`/workspace/datasets/fact-robotwin-v2/RoboTwin/robonana_norm_stats.json`，不重拟合 |
| 图像 | 当前统一 FACT resize、单图FP32 VAE、latent BF16 roundtrip，缓存/在线同一流程 |
| 精度 | FLUX/actor/Q/V FP32；冻结Qwen保持原精度 |
| action | 固定48；执行48步或提前任务终止；不做clip |
| flow采样 | Euler 20步，flow_shift=1.0 |
| 保存 | 逐帧三相机无损PNG、state/action、reward/success、真实末帧、实际seed |

固定清单发布路径（已完成，100个唯一seed）：
`/data3/hongjia/robonana/outputs/hanging_mug_fixed100_20260909/seeds.json`。
两个预检worker从200000和210000分别起选，各接受50个；实际seed以最终
清单为准，不能把候选范围当清单。预检按expert可行性筛选，不按policy成功率筛选。
这些是独立的新训练场景，不复用此前100000系列eval数据来填充本轮。

客户端按episode seed和重规划步索引生成稳定diffusion seed，避免请求排队顺序
决定噪声。后续复现还需保持指令、RoboTwin版本、task配置、机器人/资产和输入契约；
GPU batching浮点误差可能影响闭环，固定seed不等于bitwise复现。

## 3. 120k actor迁移（已完成）

原权重只读保留：
`experiments/robotwin_flux2_4b_dino_grouped_lr_A_bidir_G_causal_bs256_120k/models/checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin`。
以上及下文相对路径均相对于190代码目录。

独立导出：`checkpoints/120k_action_only_export_20260909/`，包含
`diffusion_pytorch_model.bin`、`model_config.json`、`inference_contract.json`、
`conversion_report.json`。

工具：`scripts/data/convert_120k_action_checkpoint.py`。保留FLUX、action/state投影、
actor使用的segment 0/1/2/3；丢弃旧horizon/value/DINO分支，不恢复legacy运行时。
153个同名张量复制，segment另行映射。新reward/success/Q/V头没有训练，契约
只允许action_only（每次采样1个action chunk），禁止Q排序；保存的Q候选预算32在此模式不执行。当前执行契约**不认证旧120k的历史训练输入**。
未来Stage 1需显式适配未认证预训练权重，不能绕过Stage 2/恢复的契约检查。

采样参考配置来自`experiments/hanging_mug_critic_7k_to_17k_bs16_20260908/config.json`；
只读取采样参数，没有加载该实验的FLUX或Q/V权重。旧配置的20步/shift1与参考值一致。
迁移/契约/入口32项测试通过，包括共享FACT旧/新actor相同输入噪声、多时间步的数值
一致性回归；真实4B权重已被当前服务成功加载并执行轨迹。

## 4. 各阶段脚本 / Entry points

| 阶段 | 脚本或配置 | 输入 → 输出 |
|---|---|---|
| 迁移 | `scripts/data/convert_120k_action_checkpoint.py` | 原120k → 独立action-only导出 |
| seed预检 | `scripts/internal/collect_robotwin_pool_worker.py --prepare-seeds 50 --seed-start ...` | 官方expert检查 → `accepted_seeds.json` |
| 并行采集/测速底层 | `scripts/diagnostics/benchmark_robotwin_collection_pool.py` | `--jobs-json`固定清单 → HDF5、队列、ledger、`seeds.json`、summary |
| 推理服务 | `scripts/services/inference_server_robotwin_batched.py` | checkpoint+A+图像契约 → action |
| 仿真worker | `scripts/internal/collect_robotwin_pool_worker.py` | 官方RLinf VectorEnv +当前client → 逐帧记录 |
| 回放准备 | `scripts/prepare_robotwin_rollouts.py` | HDF5 → 索引、统一图像/语言缓存，不重拟合A |
| Stage 1 | `scripts/run_robotwin_train.sh`；`robonana.configs.robotwin_flux2.config`；`ROBONANA_MAC_PHASE=world_policy` | 预训练/上轮FLUX+累积回放 → Stage-1 checkpoint |
| Stage 2 | 同一训练入口；`ROBONANA_MAC_PHASE=critic` | 本轮Stage-1 checkpoint → Q/V、Value EMA |
| 后续采集 | 同一固定清单pool，`--inference-mode action_q_rejection` | Stage-2 checkpoint → 下一轮100条 |

Round 0必须传`--inference-mode action_only`，不是把Q候选数设为1。
同一底层采集实现支持`--source-episodes`（诊断用）或`--jobs-json`（固定场景用）。
正式轮次与诊断必须使用独立输出目录，不自动把测速数据并入replay。

**现有`run_hanging_mug_mac_round.sh`尚不是本固定100场景协议的一键入口。**
它有额外action-only eval，并默认随round改变seed_group，尚未接入锁定的seed/
instruction清单。不得直接用默认调用冒充本协议。启动后续训练前需接入清单，并
验证首次Stage-1的显式未认证actor适配路径；本次没有自动启动这些阶段。

## 5. 后续训练参数（计划，未运行）

| 参数 | Stage 1 | Stage 2 |
|---|---|---|
| 新增步数 | 20,000 | 10,000 |
| FLUX | actor/world训练 | 全冻结 |
| Q/Value | 不以critic loss更新 | 训练online Q/Value |
| EMA | 无FLUX EMA | 仅Value，decay=0.995 |
| action BC | 仅成功轨迹 | 无 |
| world loss | 成功/失败都训练 | 不更新world model |
| imagination | 不做critic bootstrap | 每batch一次48步on-policy imaginary transition |
| 候选数 | — | imagination=8；环境Q选择=32 |
| gamma / reward | 0.999；非目标-1、目标0 | 相同，按逐步折扣计算chunk return |
| 执行默认值 | 每卡batch8、累积1 | 每卡batch8、累积1 |
| GPU/global batch/LR | 启动前确认并保存实际config | 启动前确认；critic默认LR=1e-4 |

池权重默认：原始成功、采集成功、历史失败、当轮失败各0.25；空池按配置重分配。
Round 0没有历史失败，不能误读成四个非空池各25%。成功尾段吸收态padding，失败
只取完整48步窗口。新critic阶段online Q/V沿用当前权重，Value EMA从当轮online
Value复制；同一阶段断点恢复才恢复原EMA/optimizer。

每次实际训练保存：Git commit、完整config、起始checkpoint、输入池清单、GPU、
总batch、LR/scheduler、步数、耗时、checkpoint路径、W&B run URL。计划不代替
运行证据；尚未启动的实验不填写虚构run ID或成功率。

## 6. 实际运行台账 / Run ledger

| ID | 实验及参数 | 产物与状态 |
|---|---|---|
| export-120k | fixed48、FP32、20步、shift1、action-only | `checkpoints/120k_action_only_export_20260909`；已完成 |
| speed-2env | GPU4推理；GPU5两个持久环境；request batch2；wait10ms；4episodes；无Q | `outputs/round0_120k_speed_2env_20260909`；814.378秒，17.682条/小时，2成功/2失败；不计入round0 |
| speed-4env | GPU4推理；GPU4/5/6/7各一个持久环境；batch2；wait10ms；同4个seed | `outputs/round0_120k_speed_4env_20260909`；359.900秒，40.011条/小时，2成功/2失败；不计入round0 |
| seed-preflight-6 | GPU6；200000起，接受50个seed | `outputs/hanging_mug_fixed100_20260909/seed_preflight_gpu6`；已完成，最后接受seed200062 |
| seed-preflight-7 | GPU7；210000起，接受50个seed | `outputs/hanging_mug_fixed100_20260909/seed_preflight_gpu7`；已完成，最后接受seed210063 |
| round-0 | 固定100训练seed；120k actor；无Q；GPU4/5/6/7各一个环境，GPU4服务batch2/wait10ms | `outputs/hanging_mug_fixed100_20260909/round0`；已启动，PID3699586；SR_0待100条完成后计算 |
| Stage 1/2 | 20k/10k | 尚未启动 |

实现提交：`cb62a66`（actor导出/能力保护）、`cac3793`（seed清单/预检）。
正式round0启动代码：`aed5c58`。190完整回归193项通过，27项Pillow弃用警告。
场景清单SHA256：`0c104e34d68c5f905a06c08c59b17e449a3570b2dbc8b99081911dff0f54af21`。
本地清单：`outputs/hanging_mug_fixed100_20260909/seeds.json`，与服务器哈希一致。
四卡实测吞吐为两卡的2.26倍。按4条小样本估算100条约2.5小时，实际受轨迹长度、
成功率及队列尾部影响；这不是硬件性能上限或完工保证。
每条采集必须核对HDF5 seed、worker ledger、队列完成状态。最终每轮逐seed结果
至少记录：round、seed、instruction、success、执行步数、耗时、HDF5路径；轮次
目录绑定明确的policy checkpoint与Git commit。逐轮原始数据独立保留、不覆盖。

## 7. Round 0实际启动参数

以下记录供审计，**当前进程已启动，不要重复执行**。输出目录拒绝覆盖。
模型Python：`/data3/hongjia/conda/envs/robonana/bin/python`。

```bash
python scripts/diagnostics/benchmark_robotwin_collection_pool.py \
  --jobs-json /data3/hongjia/robonana/outputs/hanging_mug_fixed100_20260909/seeds.json \
  --sim-gpus 4 5 6 7 --server-gpu 4 \
  --sim-python /data3/hongjia/venvs/robotwin-sapien303/bin/python \
  --robotwin /workspace/hongjia/RoboTwin \
  --checkpoint /data3/hongjia/robonana/checkpoints/120k_action_only_export_20260909/diffusion_pytorch_model.bin \
  --model-config /data3/hongjia/robonana/checkpoints/120k_action_only_export_20260909/model_config.json \
  --initial-dataset /workspace/datasets/fact-robotwin-v2/RoboTwin \
  --output /data3/hongjia/robonana/outputs/hanging_mug_fixed100_20260909/round0 \
  --inference-mode action_only --collection-round 0 \
  --inference-batch-size 2 --batch-wait-ms 10 --port 8294 --timeout-seconds 21600
```

进程环境设置`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`ROBONANA_POLICY_VERSION=120k_action_only_fixed100_round0`，PYTHONPATH包含src和
当前FACT/FLUX依赖。主管进程日志：`outputs/hanging_mug_fixed100_20260909/round0.launch.log`。
主管为独立会话后台运行，6小时安全超时；遇服务/worker异常立即停止，未完成claim
不会静默重跑。每个worker独立日志；episode完成后写盘，最终summary验证100条。

正式数据：`outputs/hanging_mug_fixed100_20260909/round0/dataset/hanging_mug/robonana_rollout/data/`。
完成结果：同round0目录下`summary.json`（success_count、success_rate、逐seed结果、耗时）。
W&B：本轮是仿真采集，当前只保存本地/服务器结构化指标；尚无训练W&B run。后续
Stage1/2启动时记录各自W&B URL，不把历史训练run当成本实验记录。
