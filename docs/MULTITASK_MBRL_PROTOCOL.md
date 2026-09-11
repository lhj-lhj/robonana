# 50-task MBRL 固定实验协议 / Fixed experiment protocol

更新：2026-09-11。单次 loop：`idx=0`。这是新的完整实验，**不续跑旧 hanging_mug 优化器或 LR 时钟**。
维护入口：[run_multitask_mbrl.py](../scripts/run_multitask_mbrl.py)。默认只打印计划，`--execute` 才执行。
算法仍是 `mac_mot_v2`，固定 chunk=48；不启用 student、softmax、Q EMA 或 EMA FLUX。

## 1. 阶段、数据、参数 / Stages

| 阶段 | 初始化 | 数据 | optimizer steps | GPU × batch × accumulation | LR：FLUX / robot |
|---|---|---|---:|---|---|
| Pretrain | 原始 FLUX.2 Klein base 4B | 原始 50 task：Clean + Randomized，预期 27,500 条成功示范 | 120,000 | 8 × 16 × 1 = 128 | 2e-5 / 1e-4 |
| Round0 | 新 `base_ckpt_120k` | 每任务 100 Clean + 100 Randomized 场景，共 10,000 个有效 policy episodes | 无训练 | 4 policy GPUs + 4 simulator GPUs | — |
| Stage1 | 新 base_ckpt_120k，重新初始化优化器/LR | 50% 原始成功示范 + 50% Round0 失败回放 | 60,000 | 8 × 16 × 1 = 128 | 2e-5 / 2e-5 |
| Stage2 | Stage1 60k；冻结整个 FLUX | 默认沿用同一 50/50 状态分布 | 20,000 | 8 × 16 × 1 = 128 | 冻结 / 1e-4 |
| Final eval | loop_ckpt_0 = Stage2 20k 中的 FLUX+Q/V | Round0 锁定的同一 50 × (100 Clean + 100 Randomized) | 无训练 | 4 policy + 4 simulator GPUs | — |

Stage2 的数据比例是本入口明确采用的默认解释，用户协议未另外指定 Stage2 配比。
训练 sampler 复用四池接口：original_success=.5，latest_failure=.5，另外两池=0。
池内按现有实现：均匀任务 → 均匀 episode → 均匀有效窗口；比例是抽样比例，不要求磁盘文件数相等。
预训练复用 EpisodeSampler，均匀 episode/window；不把每任务 50 Clean + 500 Randomized 误写成只有 Clean。
分布式分片后**单卡小批次未必恰好 8/8**，全局抽样比例仍由现有 sampler 控制，需要真实分片测试确认。

成功轨迹：action BC + image/future-state/reward/success world loss。
失败轨迹：action loss mask=0，world loss 保留。保留现有 future-state 分支与 loss 权重；没有因协议简写遗漏它而删除模型分支。
所有阶段复用当前 MAC loss 权重：image=1，action=10，future_state=.4，reward=.1，success=.1，Value/Q=1。
这不是历史旧 120k run 的逐位复现，旧权重与旧实验产物不删除。

## 2. 初始化、精度和 checkpoint

原始 FLUX 入口仅允许完整、精确的 backbone key 集合；缺 key、额外旧 robot head 或 shape 不符直接报错。
机器人输入/输出模块新初始化；专家 block/modulation 复用现有 ImageWAM transfer，scalar query/head 保持新初始化。
预训练不训练 Q/V；首次 Stage2 训练它们时仍使用 checkpoint 中的专家初始权重。
新 Stage2 的 Value EMA 从当前 online Value 复制，而不是加载上一阶段 EMA；同阶段断点续训另用已有恢复入口。

BF16 world-policy / online Q / online V；Value EMA FP32 存储和累积、BF16 autocast 前向。
return/discount/loss、加噪构造、VAE BN 反归一化保留当前 FP32 规则；冻结 Qwen 不动。
归一化所有来源固定为 A；新 replay 继续复用统一 VAE/cache 链路。

复用 FACT `checkpoint_keeps` 保留里程碑，同时每 1,000 步保存恢复点、滚动保留最近两个非里程碑点。
所有保留点保存 optimizer；不同阶段各有独立目录与 LR cosine 时钟（warmup=500）。
完成后 `aliases` 子命令验证 checkpoint 哈希/step，再创建目录符号链接，**不复制大权重**：

- Pretrain：base_ckpt_10k / 30k / 60k / 120k。
- Stage1：stage1_10k_ckpt_0 / stage1_30k_ckpt_0 / stage1_60k_ckpt_0。
- Stage2：stage2_10k_ckpt_0 / stage2_20k_ckpt_0，最后指向 loop_ckpt_0。

新协议默认全关梯度 checkpoint（八卡batch16 smoke已经通过）；`--gradient-checkpointing` 可显式恢复原策略。
显存测试使用 `--no-gradient-checkpointing --smoke-steps 3`：
必须包含真实前向、反向、Adam 更新，不以加载后显存作结论；还需给保存/可视化留余量。
若全关 OOM，保留原 checkpoint 策略，**不自动降低 batch/精度**。
NVLS 默认关闭以绕开已复现聚合问题，NVLink P2P 不关闭。

## 3. 采集与 seed 协议 / Collection

任务清单读官方 `_eval_step_limit.yml`，必须恰好 50 个；场景配置为 `demo_clean.yml`、`demo_randomized.yml`。
初始候选 seed 默认 300000，逐候选递增。场景身份为 `(task, task_config, seed)`，不同任务相同数值不算重复。

1. 独立进程内复用官方 expert feasibility check、同样的语言生成顺序，取出确切 instruction。
2. base actor **action_only** scout，每48步规划时获取3路RGB；不落盘中间图像、状态或动作。
3. scout 成功：只记 seed/instruction/结果/耗时，不保留轨迹。
4. scout 失败：同 seed + instruction + diffusion seed 重放，逐控制步记录完整三路RGB、state、action和末帧。
5. 整条执行 action 数组、长度、最终 success 必须与 scout 完全一致，才能发布失败 HDF5。
6. 成功/失败均计入初始成功率和锁定清单；重放不一致会阻止任务完成，不能按“容易重现”筛选场景。

diffusion seed 固定 `env_seed * 1000003 + control_step // 48`，并在清单记录。
每个候选含 expert/scout/replay 共用默认 1200s 墙钟预算，终止进程组额外保留清理宽限。
超时、渲染崩溃、expert 无效会记日志并换下一个候选；最多目标数量×20次尝试，超过上限显式失败。
基础设施错误不计作 policy failure。请报告替换比例：大量换 seed 会引入可运行场景的选择偏差。
**不能声称换 seed 修复了 Vulkan ErrorDeviceLost 根因**。

最终 eval 读取每任务锁定的 `seeds.json`，校验 simulator commit 和 task config 指纹，固定 instruction；
actor 32候选，Q argmax，20去噪步与 flow_shift 等从 checkpoint 契约读取。
最终 eval 用纯 `scout`，不重放、不保存图片。某个固定场景报错时记录 error 和 paired coverage，**不静默换场景**。

所有功能在 RoboNana adapter/supervisor 内维护，不改原始 RoboTwin task、物理、reward 或 action 代码。
复用关系：

- `scripts/internal/collect_robotwin_pool_worker.py`：已有 scout/replay、逐帧 writer、official expert check。
- `src/robonana/sim/collection_pool.py`：只加载固定 commit 的 RLinf/RoboTwin `robotwin/envs/vector_env.py`，不导入另一套任务实现。
  [参考源码](https://github.com/RLinf/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/robotwin/envs/vector_env.py)。
- `scripts/internal/eval_robotwin_task_isolated.py`：复用进程组终止。
- 低频RGB/同步复用部署中的 FACT `evaluation/robotwin/model2robotwin_interface.py`；不是声称 stock RoboTwin 有同名开关。

当前正式 supervisor 是**常驻 policy 服务 + 每 seed 一个隔离的模拟进程**。每个进程内仍复用官方 VectorEnv。
这是长任务防挂死优先的实现，尚未证明比跨 episode 常驻环境更快；后续吞吐测试应量化启动成本。
每个 lane 独占一张 policy GPU、一张 simulator GPU；默认4 lane。`--tasks`/`--episodes`/`--gpus` 仅用于有界测试。
只给验证通过的失败轨迹建立 `failure_dataset/.../robonana_rollout` 视图，禁止把整个 attempts 根目录直接用作训练数据。

## 4. 操作命令 / Commands

在190仓库根目录，已激活 RoboNana Python 并设置项目 PYTHONPATH 后执行。以下不含 `--execute` 的命令不会起任务。
源代码仅通过 GitHub 同步。W&B 固定团队 `hongjia-liu-aalto-university`；沿用服务器已有认证，不写入仓库。

```bash
RUN=/data3/hongjia/robonana/experiments/multitask_mbrl_v1
PY=/data3/hongjia/conda/envs/robonana/bin/python

# 预训练配置检查；正式执行需人工补 --execute。
$PY scripts/run_multitask_mbrl.py train --phase pretrain --output "$RUN"
# 独立显存 smoke，不保存 ckpt，不会接上正式训练。
$PY scripts/run_multitask_mbrl.py train --phase pretrain --output "${RUN}_gc_smoke" \
  --no-gradient-checkpointing --smoke-steps 3 --execute
# 120k完成后才创建全部里程碑链接。
$PY scripts/run_multitask_mbrl.py aliases --phase pretrain --output "$RUN" --execute

BASE="$RUN/base_ckpt_120k/transformer/diffusion_pytorch_model.bin"
BASE_CONFIG="$RUN/pretrain/config.json"
ROUND=/data3/hongjia/robonana_rollouts/multitask_round0_v1
$PY scripts/run_multitask_mbrl.py collect --output "$ROUND" \
  --checkpoint "$BASE" --model-config "$BASE_CONFIG"

# 全部收集完成、检查替换比例/重放一致性后，准备接受的失败集，仍用A统计。
$PY scripts/prepare_robotwin_rollouts.py --dataset-root "$ROUND/failure_dataset" \
  --task-glob '**/robonana_rollout' --checkpoint checkpoints/FLUX.2-klein-base-4B \
  --initial-dataset-root /workspace/datasets/fact-robotwin-v2/RoboTwin --stage all

$PY scripts/run_multitask_mbrl.py train --phase stage1 --output "$RUN" \
  --checkpoint "$BASE" --model-config "$BASE_CONFIG" --replay-root "$ROUND/failure_dataset"
# Stage1完成后同样用 aliases --phase stage1，传同样的source/replay参数。
S1="$RUN/stage1_60k_ckpt_0/transformer/diffusion_pytorch_model.bin"
$PY scripts/run_multitask_mbrl.py train --phase stage2 --output "$RUN" \
  --checkpoint "$S1" --model-config "$RUN/stage1/config.json" --replay-root "$ROUND/failure_dataset"
# Stage2完成后用 aliases --phase stage2，传同样的source/replay参数。
$PY scripts/run_multitask_mbrl.py eval --output "$RUN/final_eval" --manifests "$ROUND" \
  --checkpoint "$RUN/loop_ckpt_0/transformer/diffusion_pytorch_model.bin" \
  --model-config "$RUN/stage2/config.json"
```

生成配置不是完成训练；创建别名不是复制或重新训练。采集可读取已完成 ledger 继续剩余任务，
但异常退出时未提交的 attempt 应先人工检查隔离，不能覆盖已有半成品。训练入口拒绝覆盖已有阶段目录。
当前不自动串起三个昂贵训练阶段，也不自动取消其他GPU任务。

## 5. MAC 计算 / Existing equations

固定 n=48、imaginary H=1。policy 每状态生成8候选用于 Stage2 on-policy action selection；环境评测用32候选。
world model 对选中 action 生成下一 state/image、48个 reward logits 和 success。
reward 各步期望为 `-1 + sigmoid(logit_i)`，chunk return 为 `R = sum_i gamma^i r_i`，gamma=.999。
终止门控按当前实现的success阈值=.5；只解释目标，不改变已有算法。

`y_V = R + gamma^48 (1-d) V_EMA(s')`

`y_Q = R + gamma^48 (1-d) stopgrad(V_online(s'))`

专家直接输出归一化标量 `v=V/1000`、`q=Q/1000`；日志中的 return 是未归一化值。

`L_V = mean((v(s)-stopgrad(y_V)/1000)^2)`

`L_Q = mean((q(s,a)-stopgrad(y_Q)/1000)^2)`

Value EMA 每 optimizer update：`V_EMA ← .995 V_EMA + .005 V_online`。
FLUX/world-generated targets无梯度；Q没有EMA。公式的精确 scaling 应以 `deterministic_return_loss` 为准。

## 6. 验证门槛 / Validation gates

正式长实验前必须完成：

1. 真实原始 FLUX strict key/shape 初始化，robot 参数新初始化；分组 LR 无遗漏/重复；预训练 world-only 可反向。
2. 实际原始数据目录确有50任务、27500条，所有语言/图像缓存和A统计契约通过。不能靠glob名字判断。
3. Eight-GPU batch16 全关checkpoint至少3步，峰值显存、step时间；同配置checkpoint基线，确认没有OOM/NaN/卡collective。
4. Stage1 50/50池真实分布/失败BC mask；空失败池必须报错，禁止静默全成功训练。
5. checkpoint保存、保留、别名、Stage1→Stage2恢复与Value EMA初始化；不能仅测试配置字典。
6. Clean和Randomized分别测成功/失败scout重放；核对action逐元素、RGB数、末帧、instruction和success，不一致就不接受该回放。
7. 注入单seed超时/崩溃、确认清理与换seed；重启ledger不重复计数；final eval错误不改变locked清单。
8. 八卡四lane实测吞吐/初始化时间/峰值RAM与显存；样本跨短/长任务，不从两个hanging_mug样本外推全50task速度。

注意：失败数据只包含base policy失败分布，稀有失败任务可能没有样本；50/50 oversampling不等于泛化得到保证。
scout+replay加速取决于SR：失败越多、重放开销越大，不保证始终比逐帧直接采集快。
初始与最终报告均应按task×clean/random给SR、有效数、error数、seed替换率，再给50任务宏平均。

## 7. 本次执行记录

2026-09-11：按用户要求停止旧 Q/V Stage2 `hanging_mug_critic4000_4gpu_acc2_r2_20260910`。
最后观察日志step7930；最新完整checkpoint为 `checkpoint_epoch_23_step_7000`，包含FLUX/专家、Value EMA和optimizer。
通过终止已核实的 accelerate launcher PID992629 完成退出，未停止其他任务。尾段未另存。
新协议的长训练、10000场景采集尚未启动。测试结果在完成后补入本节；未运行的门槛不标作通过。

- 190首轮相关回归61项通过；新增seed故障回归后协议文件7项通过（与前61项有重叠，不相加）。
- 真实数据元信息：Clean 50任务×50=2500；Randomized 50任务×500=25000。
- 首次八卡、batch16、全关GC smoke在真实数据契约检查失败：`Clean/adjust_bottle/flux_cache/latents_v2/_contract.json` 缺失。
  已进入prepare，原始FLUX与optimizer准备未报错，但没有完成训练step，因此这次测试不能给出显存可行性结论。
- 日志：`outputs/multitask_protocol_validation_20260911/gc_off.log`；W&B run `qozrly7w`，用户团队正确。
- 仅显存探针允许 `--smoke-task-globs Clean/hanging_mug --smoke-steps 3`。正式入口拒绝无smoke预算的数据子集覆盖；
  这只能测试相同tensor尺寸下的显存，不能认证50任务缓存已经可用。
- 子集八卡3-step全关GC完成：峰值allocated=101.211 GiB、reserved=107.293 GiB（所有rank最大值）；
  首步5.549s，随后0.9268/0.9285s；均完成优化器更新、无OOM/NaN。此probe没有测checkpoint保存峰值。
  日志 `outputs/multitask_protocol_validation_20260911/gc_off_subset.log`，W&B `l9ojo93a`。
- 缓存覆盖：100个task/config目录中仅 `Clean/hanging_mug` 存在latents_v2契约，99个尚缺。
  缺失缓存准备使用现有入口（此处记录命令，尚未启动全量预处理）：

```bash
PYTHONPATH=src:third_party/FACT:third_party/flux2/src:third_party/flux2_official/src \
/data3/hongjia/conda/envs/robonana/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/data/preprocess_robotwin_lerobot_flux.py \
  --dataset-root /workspace/datasets/fact-robotwin-v2/RoboTwin \
  --checkpoint checkpoints/FLUX.2-klein-base-4B \
  --task-glob 'Clean/*' --task-glob 'Randomized/*' --stage images
```

这一步是新链路实际编码，不是重命名旧缓存。应先做1个episode的有界转换/一致性和耗时测试，
估算全量磁盘与时间后再开27500条。语言缓存完整性也需要独立检查。
