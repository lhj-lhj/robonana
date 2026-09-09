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
新增 optimizer steps，不是终身 checkpoint 步数。Round0已完成；Stage1八卡训练已正常更新并上传W&B，Stage2未启动。

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
本次Stage1显式适配未认证预训练权重（见第9节），不绕过Stage2/恢复的契约检查。

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
| Stage 1 | `scripts/run_robotwin_train.sh`；`robonana.configs.robotwin_flux2_4b_mac.config`；`ROBONANA_MAC_PHASE=world_policy` | 预训练/上轮FLUX+累积回放 → Stage-1 checkpoint |
| Stage 2 | 同一训练入口；`ROBONANA_MAC_PHASE=critic` | 本轮Stage-1 checkpoint → Q/V、Value EMA |
| 后续采集 | 同一固定清单pool，`--inference-mode action_q_rejection` | Stage-2 checkpoint → 下一轮100条 |

Round 0必须传`--inference-mode action_only`，不是把Q候选数设为1。
同一底层采集实现支持`--source-episodes`（诊断用）或`--jobs-json`（固定场景用）。
正式轮次与诊断必须使用独立输出目录，不自动把测速数据并入replay。

**现有`run_hanging_mug_mac_round.sh`尚不是本固定100场景协议的一键入口。**
它有额外action-only eval，并默认随round改变seed_group，尚未接入锁定的seed/
instruction清单。不得直接用默认调用冒充本协议。当前Stage1由独立训练入口显式
启动（见第9节）；整轮自动化仍需接入固定清单，本次不自动启动Stage2或下一轮采集。

## 5. 两阶段训练参数（Stage1已运行，Stage2为计划）

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
| 本实验执行配置 | 每卡batch32、累积1 | 默认每卡batch8、累积1；尚未启动 |
| GPU/global batch/LR | GPU0–7；global256；LR=2e-5 | 启动前确认；critic默认LR=1e-4 |

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
| round-0 | 固定100训练seed；120k actor；无Q；GPU4/5/6/7各一个环境，GPU4服务batch2/wait10ms | `outputs/hanging_mug_fixed100_20260909/round0`；已完成100条，41成功/59失败，SR_0=41%；5309.017秒（1小时28分29秒），67.809条/小时；完整性校验通过 |
| Stage 1 | 20k；GPU0–7；每卡32、累积1、global256；100回放+50Clean | `experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_20260909`；已正常运行；W&B确认收到_step33指标，total_loss=1.083（启动快照） |
| Stage 2 | 10k | 未启动；不自动串联 |

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

以下记录供审计，**该次采集已经完成，不要重复执行**。输出目录拒绝覆盖。
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

## 8. 数据选择与采样方式（按现有实现核对）

本轮数据范围定为**100条round0真实回放 + 50条原始Clean示范**，共150条，
其中成功91条（50原始+41采集），失败59条。不添加500条Randomized。
理由：本实验首先测量固定clean场景上的RL loop；同时改变数据量与背景分布会
引入另一个变量。100+550可以作为后续独立对照，但不能断言一定更好或更差。

### 8.1 训练窗口采样：不是把150条轨迹直接等概率混在一起

复用现有`src/robonana/data/robotwin_hdf5.py::RoboTwinPosttrainSampler`，
配置来自`src/robonana/configs/posttrain_config.py`；本次没有新增或修改采样器。
默认四池各占0.25，round0的历史失败池为空，其权重转到当轮失败池，因此：

| 数据池 | 本轮轨迹数 | 实际采样占比 | sampler batch=8举例 |
|---|---:|---:|---:|
| 原始Clean成功示范 | 50 | 25% | 2个窗口 |
| 累积采集成功 | 41 | 25% | 2个窗口 |
| 历史失败（round<0） | 0 | 0% | 0 |
| 当轮失败（round=0） | 59 | 50% | 4个窗口 |

每个sampler batch先按池比例分配窗口数；不整除时向下取整，再按余数大小补齐。
池内依次**均匀选任务 → 均匀选episode → 均匀选该episode的合法起始帧**。
本实验只有hanging_mug一个任务。采样有放回，不要求一轮遍历每条轨迹一次；
不会因为失败轨迹更长就按全数据帧数占据更大比例。batch内再shuffle。
表中batch指采样器收到的batch_size，实际多卡分片以启动配置为准。

成功轨迹允许末尾吸收态padding；失败轨迹只提供完整48步窗口，不padding。
Stage1的action BC仅使用成功窗口；world loss使用所有窗口。Stage2复用同一
数据池采样逻辑，但其训练action由当前policy生成，不是直接拿真实action做BC。
后续历史失败池非空时恢复四池各25%；成功池为空则其权重转给原始成功池。

### 8.2 Policy候选采样与Q选择

对每个当前观测，独立噪声`z_i ~ N(0,I)`，用flow policy产生48步action chunk：
`a_i = FlowPolicy(language, state, current_image, z_i)`，20步Euler、shift=1。

- Round0：只采样一个chunk，直接执行；不计算Q，不做argmax，也没有softmax。
- Stage2 imagination：当前配置采样8个候选，计算Q并取`argmax_i Q(s,a_i)`；
  对选中chunk生成一次48步world-model转移，用于Q/V bootstrap训练。
- 后续环境采集：采样32个候选，计算Q并取argmax，执行选中chunk；任务提前成功
  则停止，否则执行满48步后重新观测、规划。不是一次选择整条episode。
- request batch=2是两个环境请求并行；candidate batch是候选计算分组，均不改变
  每个环境的总候选数。Round0未使用候选分组进行Q搜索。

### 8.3 启动顺序检查

Round0完成时只有action-only导出；该导出的reward/success/Q/V头未训练。
必须先完成本实验Stage1（20k）或明确指定另一个已验证的world-policy checkpoint，
才能启动Stage2（10k）。不能因100条数据已收集，就跳过world-model学习直接冻结
120k转换模型做imaginary bootstrap。历史其它数据/模型的Stage1不能自动算作本实验Stage1。

## 9. Stage1实际配置与启动检查（2026-09-09）

用户指定：100条round0回放 + 50条原始Clean，8卡、每卡batch32、累积1、global
batch256、20,000步。复用`run_robotwin_train.sh`及`robotwin_flux2_4b_mac.config`，
没有另建训练框架；Stage2不自动启动。

| 参数 | 实际设置 |
|---|---|
| GPU / batch | 0,1,2,3,4,5,6,7；32×8×1=256 |
| optimizer / LR | AdamW；2e-5；betas=(0.9,0.95)，eps=1e-8，weight_decay=1e-4 |
| scheduler | warmup500步，cosine decay到20,000步 |
| phase / round | world_policy / 消费round0数据 |
| 初始化 | `checkpoints/120k_action_only_export_20260909`，新阶段，不恢复旧optimizer/步数 |
| 显式适配 | `ROBONANA_ALLOW_UNCERTIFIED_PRETRAIN=1`，`ROBONANA_RESUME=0` |
| 原始任务筛选 | `ROBONANA_POSTTRAIN_ORIGINAL_TASK_GLOBS=Clean/hanging_mug` |
| 回放根目录 | `outputs/hanging_mug_fixed100_20260909/round0/dataset` |
| 精度/内存 | FP32；模型gradient_checkpointing开启；DeepSpeed ZeRO-2 |
| 数据worker | 每进程4；OMP/MKL线程各1 |
| loss权重 | action10，image1，future_state0.4，reward0.1，success0.1 |
| checkpoint | 第100步早期保存，之后每1000步；保留数量1，保存optimizer |
| W&B | online；name=`hanging_mug_fixed100_round0_stage1_20k_bs32x8_20260909`；[run pkkg5pyx](https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/pkkg5pyx) |

输入检查已完成：四池轨迹数`[50,41,0,59]`，统一A统计。回放65,931帧的当前VAE/
语言缓存校验通过，原始50条Clean也已生成当前缓存；旧`latents`未删除、不重标。
复用的准备入口：`prepare_robotwin_rollouts.py`生成/验证索引，
`data/preprocess_robotwin_lerobot_flux.py`与`data/preprocess_robotwin_flux.py`通过
八卡torchrun分片处理，后者开启`--per-episode-language`。未处理Randomized。

代码调整（`4db0a12`）：

1. 在现有inference_contract增加显式新Stage1加载校验；仍检查权重指纹、采样参数、
   图像契约、归一化。converted actor只在allow开启且resume关闭的world_policy阶段允许。
2. Trainer复用该校验及既有`initialize_scalar_expert_from_flux`，为转换actor的
   Q/Value主体按ImageWAM规则从FLUX初始化；query/head保留新初始化，actor不改。
   既有已训练MAC checkpoint不重新初始化expert。
3. 增加Stage2拒绝、resume拒绝、未授权拒绝、统计不一致与权重篡改拒绝测试。
   190完整回归194项通过；仅有27条已有Pillow弃用警告。

启动诊断：首轮八卡进程卡在`init_process_group → eager_connect_single_device`，
由只读Python调用栈确认，尚未进入模型前向，不能归因为batch32 OOM。
重试1设置`NCCL_IB_DISABLE=1`并开启INFO日志，仍未完成初始化；重试2额外关闭
可选NVLS聚合（`NCCL_NVLS_ENABLE=0`），保留NVLink P2P。未改batch、精度或数据。
各次日志为`outputs/hanging_mug_fixed100_20260909/stage1_train*.launch.log`，全部保留。
W&B现有凭据通过官方HTTP API验证；SDK public API曾报告relogin required。
训练重试显式从服务器已有netrc读取密钥到进程环境，不输出密钥、不写入配置或Git。
训练run已创建；官方API返回state=running并收到_step33、total_loss=1.083，确认在线
上传成功，不是仅打印了本地run地址。只读临时诊断工具未安装进训练环境。

### 启动验证结果

重试2关闭NVLS后NCCL完成初始化，进入实际forward/backward/optimizer更新。
前10步loss均有限，无OOM；稳定单步约4.3–4.4秒、吞吐约59个窗口/秒。
八卡显存快照76,081–78,755MiB（约74.3–76.9GiB/卡）；20k预计约24小时，另加
checkpoint保存与后续运行波动，不是完工保证。未通过降低batch或精度实现启动。
实际配置JSON已核对：gpu_ids=0–7、batch_size_per_gpu=32、accumulation=1、
max_steps=20000、resume=false。模型源码commit为`4db0a12`。
有效训练日志：`outputs/hanging_mug_fixed100_20260909/stage1_train_retry2.launch.log`。
启动主管PID3778762；首轮/重试1仅初始化失败、无optimizer更新，日志独立保留。
本次不更新系统驱动/Fabric服务，不关闭NVLink P2P，不自动启动Stage2。

### 9.1 关闭梯度检查点试跑：OOM，当前暂停（2026-09-09）

用户要求关闭梯度检查点，保持每卡32、八卡、累积1、FP32不变。
原训练在step558中断，最后完整checkpoint仍为step100；没有即时保存接口，
101–558步未保存，不能声称从558续训。原目录、日志及step100的模型/Adam/
scheduler/RNG文件均保留，不覆盖。总目标仍20,000步，不另加20k。

复用FACT恢复、现有模型`disable_gradient_checkpointing()`及配置复制工具，
增加`world_policy_resume.config`配置适配器；未修改forward、mask、loss、
模型结构或Stage2算法。与critic延长训练不同，此入口不改变卡数、batch、
数据、精度、学习率曲线或预算，也不重置expert。只对新配置设置
`gradient_checkpointing=False`，指定恢复路径及新的W&B运行。

实际入口（以下是已执行记录，**不要自动重复启动**）：

```bash
export ROBONANA_PROJECT_DIR=/data3/hongjia/robonana/experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_nogc_20260909
export ROBONANA_RESUME_CONFIG=/data3/hongjia/robonana/experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_20260909/config.json
export ROBONANA_RESUME_CHECKPOINT=/data3/hongjia/robonana/experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_20260909/models/checkpoint_epoch_1_step_100
export ROBONANA_PYTHON=/data3/hongjia/conda/envs/robonana/bin/python
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/run_robotwin_train.sh --config robonana.configs.world_policy_resume.config
```

W&B继续使用服务器已有netrc凭据注入环境，不写入Git；新run为
[a924f45f](https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/a924f45f)。
完整日志：`outputs/hanging_mug_fixed100_20260909/stage1_nogc.launch.log`。
运行代码commit：`85adbe3`。Windows配置测试3项通过，190配置/恢复测试6项通过。

实测结果：八卡NCCL初始化通过，模型日志确认`gradient_checkpointing=False`，
DeepSpeed模型与优化器恢复成功；首个训练batch尚在前向single block/RoPE处即OOM，
未完成新的optimizer step。报错显示每卡进程占用约178.17GiB，PyTorch实际分配
173.95GiB、空闲仅157.56MiB，新增446MiB分配失败。不能把这次失败归为NVLS，
也不能仅用关闭前64–67GiB的快照判断关闭后的峰值足够。未降低batch/精度，
未自动重启或恢复梯度检查点；当前训练暂停，等待选择部分block重计算等方案。

### 9.2 NVLS进一步定位：Fabric Manager组播状态错误

在原训练已停止、新训练未启动的间隙，用同一Python/PyTorch/NCCL环境做有界
最小复现，只初始化进程组并AllReduce一个FP32张量，不加载RoboNana。
每组最多45秒，超时只终止该组诊断子进程，不重置GPU、不改系统服务。

| 测试 | 结果 | 实际NVLS路径 |
|---|---|---|
| GPU6/7，两卡，NVLS_ENABLE=1 | 通过，AllReduce=2 | **0 nvls channels**，实际跳过NVLS |
| GPU0–7，八卡，NVLS_ENABLE=1 | 45秒内未完成初始化，终止 | 创建组播组后停留在共享句柄导入附近 |
| GPU0–7，八卡，NVLS_ENABLE=0 | 通过，AllReduce=8 | P2P/CUMEM，NVLink P2P保留 |

两卡通过不能称为“NVLS通过”：匹配版本NCCL 2.27.3的
[`src/transport/nvls.cc::ncclNvlsInit`](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/transport/nvls.cc)
明确在`gpuCount <= 2`时返回，不创建NVLS通道。因此此前两卡训练正常并不矛盾。
旧120k八卡训练日志缺少对应NCCL初始化证据，不能推定它实际启用了NVLS。

服务器`/var/log/fabricmanager.log`在2026-09-09 05:10:02 UTC（本次最小复现）
以及04:11:49、04:21:38（两次正式启动失败）均有同类错误：

```text
requesting GPU handle 0x0 is different from exporter GPU handle ...
cannot find exporter GPU in partition Id 57082 gpuHandle 0x0 ...
All GPUs in the partition need to be reset to recover
failed to add multicast team ...
```

同类错误至少在本次开机后的9月1日、4日、6日已出现；不是本次改batch或模型后
才产生。Fabric服务显示active，nvidia-smi Fabric显示Completed/Success/Healthy，
但这些摘要不足以证明NVLS组播分配正常。已定位到Fabric组播请求的GPU句柄/
分区状态错误；尚未证明最初由哪次驱动/服务事件造成，不等于确定某个NCCL补丁可修复。
系统日志建议分区GPU reset；这属于影响整机GPU作业的维护操作，本次没有执行。

诊断原始日志：`outputs/hanging_mug_fixed100_20260909/nvls_probe_20260909/`。
后续恢复NVLS需先协调停掉整个分区GPU任务，再由管理员按平台维护流程处理，
随后重跑最小通信测试与性能对照；不要在正式训练中直接开启或重启Fabric服务。

### 9.3 部分block检查点续训（2026-09-09）

用户最终要求保持每卡32、8卡、累积1、global256、FP32，不采用batch16方案。
复用9.1的Stage1恢复入口，增加single block重计算stride；没有另建训练器或
改动attention/mask、loss、权重结构。默认stride1仍保留原先全block检查点行为。
此实验stride2：5个double block全部checkpoint；20个single block中只对
0、2、4、6、8、10、12、14、16、18号checkpoint，关闭另外10个的重计算。
训练外eval不使用checkpoint；模型state_dict不增加权重或buffer键。

相对9.1启动命令，替换project目录为
`experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_partialgc_20260909`，并设置：

```bash
export ROBONANA_GRADIENT_CHECKPOINTING=1
export ROBONANA_GRADIENT_CHECKPOINTING_SINGLE_STRIDE=2
```

其余来源仍是原八卡实验的完整step100，目标20k（不是从558恢复，也不是另加20k）。
保留原数据100回放+50Clean、A统计、优化器状态及原warmup/cosine计划。
继续使用`NCCL_NVLS_ENABLE=0`，NVLink P2P保留；没有reset GPU或重启Fabric服务。
W&B：[3ca5f86f](https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/3ca5f86f)。
日志：`outputs/hanging_mug_fixed100_20260909/stage1_partialgc.launch.log`。
启动源码commit：`1a3ff3b`；主管PID3829549。

验证新增：同一小型MAC模型上全关/全开/部分开启的5类world-policy输出逐元素一致，
所有有梯度参数的梯度在rtol1e-5/atol1e-6内一致；检查checkpoint调用数、eval不重计算、
非法stride拒绝、配置复制不改batch/数据/scheduler。190针对性7项测试通过。

启动实测已完成真实forward/backward/optimizer更新，截至step123未见OOM/非有限loss。
稳定单步中位数约4.015秒（step111–123），原全检查点最后50步中位数4.3885秒：
单步耗时约减少8.5%、吞吐约增加9.3%。这是先后运行的短窗口比较，不是硬件极限
或严格同batch样本A/B；启动首步11.6秒不计入稳定速度。按4秒/step估计剩余约22小时，
另加保存与运行波动。训练保持运行，没有因测试结束停止。

八卡nvidia-smi显存快照均177,045MiB（172.9GiB），卡总显存183,359MiB，
可见余量约6.2GiB；这是进程/驱动显存快照，不是逐算子allocated峰值。
因此不继续扩大关闭范围；不能将少重计算40%的block解释为提速40%。
实际config.json再次核对：GPU0–7、每卡32、累积1、FP32、stride2，目标20k。
190完整回归196项通过（122.70秒，27条已有Pillow弃用警告）。W&B官方API已确认
run为running，收到_step150、total_loss=0.5543、samples_per_sec=63.79，非仅本地打印URL。

### 9.4 当前 BF16 仓库配置重启与全关 checkpoint 实测（2026-09-09）

中文：本次只调整实验启动参数，复用 `world_policy_resume` 和 FACT 恢复路径，
没有修改算法、训练器或新增精度兼容分支。启动源码为 `81d7633`。
English: Runtime-only restart using the maintained resume adapter; no algorithm
or trainer changes. Full activation-checkpoint disabling was tested, not assumed safe.

原 FP32/partial-GC 进程最终停在 step3179，最新完整保存点为 step3000；
3001–3179 未保存，需要重跑。原 checkpoint、日志和数据全部保留。
用户明确确认后停止占用 GPU0–5 的另一组 ImageWAM 训练，仅发送中断信号，
未删除其产物；随后确认八卡空闲再启动本实验。

恢复源：
`experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_partialgc_20260909/models/checkpoint_epoch_10_step_3000`。
旧 FP32 DeepSpeed 优化器文件名为 `zero_pp_rank_*_mp_rank_00_optim_states.pt`，
BF16 恢复要求 `bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt`。先核对全部八片
ZeRO-2 状态：partition_count=8、Adam step=3000、主权重及两个动量为 FP32。
再创建独立硬链接视图
`experiments/checkpoint_views/partialgc_step3000_bf16_names_20260909`，保留所有原文件，
仅增加 BF16 文件名别名，不改内容、不覆盖源 checkpoint。该视图只用于读取恢复，
不得原地写入文件。真实八卡恢复日志确认模型、Adam、scheduler、sampler 和 RNG 加载成功。
这是同拓扑的 FP32→BF16 续训，不是数值完全等价的历史重现，也不是八卡转两卡的重分片。

全关试跑 `..._bs32x8_bf16_nogc_r2_20260909`：
`gradient_checkpointing=False`，每卡32、累积1、global256、BF16，首个前向即 OOM。
报错时每卡进程约178.07GiB，PyTorch allocated=172.52GiB、reserved但未分配=3.62GiB，
仅余约253.56MiB，再申请1.96GiB失败。没有完成新 optimizer step。
日志：`outputs/hanging_mug_fixed100_20260909/stage1_bf16_nogc_r2.launch.log`。
此前无文件名别名的首次 BF16 启动仅在恢复时报 missing optimizer filename，
不能把该次失败算作显存测试。

正式回退到部分 checkpoint：5个double全部保留，20个single中偶数编号保留。
batch、数据、学习率及终点不变：GPU0–7，每卡32，累积1，global256；
100条回放+50条原始Clean、A统计；lr=2e-5、原warmup/cosine，终点20000。
NVLS仍关闭，NVLink P2P保留。使用与9.1相同的正式入口，覆盖如下环境参数：

```bash
export ROBONANA_PROJECT_DIR=/data3/hongjia/robonana/experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_bf16_partialgc_20260909
export ROBONANA_RESUME_CONFIG=/data3/hongjia/robonana/experiments/hanging_mug_fixed100_round0_stage1_20k_bs32x8_partialgc_20260909/config.json
export ROBONANA_RESUME_CHECKPOINT=/data3/hongjia/robonana/experiments/checkpoint_views/partialgc_step3000_bf16_names_20260909
export ROBONANA_GRADIENT_CHECKPOINTING=1
export ROBONANA_GRADIENT_CHECKPOINTING_SINGLE_STRIDE=2
export ROBONANA_PYTHON=/data3/hongjia/conda/envs/robonana/bin/python
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
bash scripts/run_robotwin_train.sh --config robonana.configs.world_policy_resume.config
```

已完成真实更新至step3037；最近30步单步中位数2.1055秒，约121.6 samples/s。
相较重启前FP32约4.0秒/step，吞吐约1.9倍；这是短窗口先后比较，不是严格A/B。
八卡显存快照均101877MiB（99.49GiB），总183359MiB，余量79.57GiB；
这是nvidia-smi进程/驱动快照，不是逐算子allocated峰值。全关OOM已实测，不能
由部分checkpoint的显存余量推断全关一定可行。按当前速度剩余约10小时，不含保存开销。
W&B服务端确认running，已收到step3032、total_loss=0.335209、samples_per_sec=121.276。
W&B：[ob7ztgvt](https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/ob7ztgvt)。
日志：`outputs/hanging_mug_fixed100_20260909/stage1_bf16_partialgc.launch.log`。
启动主管PID232441；训练保持运行。
