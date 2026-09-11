# 50-task MBRL 固定实验协议 / Fixed experiment protocol

更新：2026-09-11。单次 loop：`idx=0`。这是新的完整实验，**不续跑旧 hanging_mug 优化器或 LR 时钟**。
维护入口：[run_multitask_mbrl.py](../scripts/run_multitask_mbrl.py)。默认只打印计划，`--execute` 才执行。
`audit` 是例外：只运行真实数据元信息、池非空、A统计和cache契约校验，不分配训练模型、不启动仿真。
正式 `train --execute` 也会先执行这项CPU preflight，避免等八份FLUX加载之后才发现缓存缺失。
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
$PY scripts/run_multitask_mbrl.py audit --phase pretrain --output "$RUN"
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
- 最新相关回归集合62项通过（52.90s）；涵盖真实配置构造、原始FLUX初始化、分组LR、scout/replay队列、
  跨进程超时清理、首次收集换seed及锁定eval不换seed。没有把这些单测当作10,000场景压力测试。
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

### 有界采集 smoke（不是正式 Round0）

使用已有 `120k_action_only_export_20260909`，GPU6 policy/GPU7 simulator，
`hanging_mug`、seed200001、每配置目标1条、最多1个候选、每候选300s。
Clean expert检查通过；scout执行900步失败，进入逐帧重放；重放未在此严格预算内完成，
collector触发 `TimeoutError: collection probe exceeded its bounded deadline`。
supervisor写入 infrastructure_error，不计作有效policy episode，测试候选预算耗尽后退出。
没有发布HDF5（文件数0），八卡均已释放；Randomized未进入，所以不能报“clean/random重放测试通过”。
首次收集换新seed、最终locked eval不换seed由故障注入测试覆盖，真实多seed替换还要进一步验收。
正式默认1200s不是该smoke的300s；尚未据此测出8卡吞吐或完成50任务压力测试。
产物：`outputs/multitask_protocol_validation_20260911/infra_smoke/`，包含任务ledger、命令配置和独立服务/环境日志。

本次只改RoboNana配置/初始化适配/采集总控及文档测试；没有改RoboTwin任务源代码或RoboNana的MAC目标公式。
旧单任务实验MD和posttrain_config中用户已有未提交修改保持原样。

## 8. 全量缓存生成与正式训练命令（2026-09-11）

### 为什么以前生成过，现在又要生成？

旧缓存没有删除：`flux_cache/latents` 仍有27,500个文件，合计417.19 GiB。
语言 `flux_cache/language` 仍有27,500个文件，合计805.65 GiB；本次不重算Qwen、不改其精度。
旧 `_manifest.json` 只记录shape/dtype，不证明与后来统一的在线编码数值一致。
2026-09-08的提交 `1c83eb0` 将在线/缓存统一成 FACT像素变换、FP32单图VAE、禁TF32、BF16落盘，
新缓存改用 `latents_v2` 并附VAE/运行库指纹。此前只有Clean/hanging_mug的50条完成v2转换。
因此“缺99个配置的缓存”准确说是缺**新链路缓存**，不是原缓存消失。

全数据实际为6,075,103帧，新缓存纯tensor约417.14 GiB；启动前/data3剩余约2.7 TiB。
旧缓存、旧checkpoint和所有原始视频都保留；不使用`--overwrite`，重复执行会跳过有效v2文件。
目录的 `_contract.json` 表示编码规范，不表示该目录已经全量完成。
训练入口已加强：每个episode必须有有效图像文件/完成证明及正确shape/dtype的语言文件，避免缓存生成一半就误开训。

### 一次性环境设置 / Environment

以下命令都在190执行；先设置这些变量，再复制后续命令。所有输出都在190，本机同名路径不是服务器目录。

```bash
cd /data3/hongjia/robonana
export PYTHONPATH="$PWD/src:$PWD/third_party/FACT:$PWD/third_party/flux2/src:$PWD/third_party/flux2_official/src"
PY=/data3/hongjia/conda/envs/robonana/bin/python
DATA=/workspace/datasets/fact-robotwin-v2/RoboTwin
FLUX="$PWD/checkpoints/FLUX.2-klein-base-4B"
CHECK="$PWD/outputs/cache_full_validation_20260911"
RUN="$PWD/experiments/multitask_mbrl_v1"
ROUND=/data3/hongjia/robonana_rollouts/multitask_round0_v1
mkdir -p "$CHECK"
```

### 单episode转换与实际落盘一致性 / Completed pilot

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/data/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "$DATA" --checkpoint "$FLUX" --task-glob Clean/adjust_bottle \
  --stage images --max-episodes 1

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/diagnostics/verify_image_pipeline.py \
  --checkpoint "$FLUX" --lerobot-task "$DATA/Clean/adjust_bottle" --device cuda:0 \
  --verify-saved-cache --report "$CHECK/single_parity.json"
```

结果：episode0共143帧，转换段耗时约3.69s；缓存shape `[143,288,128]`，全tensor有限。
抽查帧0/1：实际落盘缓存、offline编码、online batch1/batch2完全一致，max_abs_error=0。
这是两帧严格数值对照加整episode结构/有限值检查，不是宣称逐帧全部重新编码比对。
相关测试24项通过。单episode速度不能直接当全数据吞吐结论。

### 八卡全量缓存 / Active job

**2026-09-11已启动，下列命令是复现/断点重跑入口，不要在现有任务运行时重复启动。**
启动commit `1ce717e`；torchrun PID `1739983`，PID会失效，以实际进程和日志为准。
此次应复用51份有效v2文件，生成27,449份；不会生成新的语言缓存。

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/data/preprocess_robotwin_lerobot_flux.py \
  --dataset-root "$DATA" --checkpoint "$FLUX" \
  --task-glob 'Clean/*' --task-glob 'Randomized/*' --stage images \
  > "$CHECK/full_cache.log" 2>&1 < /dev/null &
echo $! > "$CHECK/full_cache.pid"

tail -n 20 "$CHECK/full_cache.log"
# 全部worker正常退出之后验收；此命令不会起训练：
"$PY" scripts/run_multitask_mbrl.py audit --phase pretrain --output "$RUN"
```

`--batch-size 64`（默认）只是解码/传输分组，VAE内部仍逐图FP32编码，不能为了吞吐擅自改成批量卷积。
终端断开不会取消任务。重跑前核实旧torchrun和8个worker均已退出；不要同时开两个生成器写同一个cache。
验收必须达到27,500个有效episode，不能仅看到100个契约文件就认为完成。

### 正式预训练、Stage1、Stage2 / Explicit training launches

以下是**人工正式启动**命令，本轮没有执行。W&B仍使用服务器已有认证和用户团队。
默认八卡×16×累积1，BF16；梯度checkpoint全关；不要与缓存/验收任务抢同一批GPU。

```bash
# 原始FLUX开始，120k，backbone LR2e-5 / robot LR1e-4。
nohup "$PY" -u scripts/run_multitask_mbrl.py train --phase pretrain --output "$RUN" --execute \
  > "$CHECK/pretrain.launch.log" 2>&1 < /dev/null &

# 预训练完成后，生成10k/30k/60k/120k里程碑别名。
"$PY" scripts/run_multitask_mbrl.py aliases --phase pretrain --output "$RUN" --execute
BASE="$RUN/base_ckpt_120k/transformer/diffusion_pytorch_model.bin"

# 必须先完成该base policy的正式Round0采集及失败集缓存。
"$PY" scripts/run_multitask_mbrl.py collect --checkpoint "$BASE" \
  --model-config "$RUN/pretrain/config.json" --output "$ROUND" --execute
"$PY" scripts/prepare_robotwin_rollouts.py --dataset-root "$ROUND/failure_dataset" \
  --task-glob '**/robonana_rollout' --checkpoint "$FLUX" --initial-dataset-root "$DATA" --stage all

# Stage1，60k，全部LR2e-5。
"$PY" scripts/run_multitask_mbrl.py train --phase stage1 --output "$RUN" \
  --checkpoint "$BASE" --model-config "$RUN/pretrain/config.json" \
  --replay-root "$ROUND/failure_dataset" --execute
"$PY" scripts/run_multitask_mbrl.py aliases --phase stage1 --output "$RUN" \
  --checkpoint "$BASE" --model-config "$RUN/pretrain/config.json" \
  --replay-root "$ROUND/failure_dataset" --execute
S1="$RUN/stage1_60k_ckpt_0/transformer/diffusion_pytorch_model.bin"

# Stage2，20k，Q/V LR1e-4，FLUX冻结，仅Value有EMA。
"$PY" scripts/run_multitask_mbrl.py train --phase stage2 --output "$RUN" \
  --checkpoint "$S1" --model-config "$RUN/stage1/config.json" \
  --replay-root "$ROUND/failure_dataset" --execute
"$PY" scripts/run_multitask_mbrl.py aliases --phase stage2 --output "$RUN" \
  --checkpoint "$S1" --model-config "$RUN/stage1/config.json" \
  --replay-root "$ROUND/failure_dataset" --execute
```

### 缓存完成后的授权验收 / Pending, not yet scheduled

先检查GPU占用，只使用空闲卡，不取消其他任务。测试输出不能混进正式Round0数据。

```bash
# 小模型、真实八进程DeepSpeed ZeRO：检查冻结参数、Q/V、FP32 Value EMA、Adam、LR和RNG恢复。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_NVLS_ENABLE=0 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=8 scripts/diagnostics/validate_mac_distributed_safety.py \
  --backend deepspeed --mode resume --checkpoint-dir "$CHECK/tiny_ds8_resume"

# 现有120k actor仅用于infra测试；四个任务×clean/random×2 episodes，共16个目标episode。
# 4 policy GPUs + 4 simulator GPUs；给完整失败重放1200s/seed预算。
"$PY" scripts/run_multitask_mbrl.py collect --output "$CHECK/infra_eight_gpu" \
  --checkpoint "$PWD/checkpoints/120k_action_only_export_20260909/diffusion_pytorch_model.bin" \
  --model-config "$PWD/checkpoints/120k_action_only_export_20260909/model_config.json" \
  --tasks hanging_mug move_stapler_pad place_mouse_pad stamp_seal \
  --episodes 2 --gpus 0 1 2 3 4 5 6 7 --seed-start 300000 \
  --seed-timeout 1200 --candidate-multiplier 3 --execute
```

要求Clean和Randomized都实际覆盖失败重放，检查action_exact、replay_verified、末帧和有效transition；
若没有失败样本，只能报“未覆盖”，不能判重放一致性通过。
吞吐分别报告端到端（含expert/启动/重放）和scout用时，不用单纯GPU利用率判断性能上限。
小模型DS恢复测试不等于4B模型完整保存/恢复测试；4B实测继续复用现有真实训练/恢复入口，另行记录。

### 当前运行记录与接续状态

#### 2026-09-12 验收接续

- 用户已授权定时接续，全部验收通过后启动正式120k预训练；自动化 `robonana` 已创建。
  下方早期“未创建自动化”的记录是历史状态，不代表当前授权。
- 全部8个缓存worker完成并释放GPU。实际逐episode audit通过：27,500条、6,075,103帧，
  `per_episode_files_verified=true`，包括元数据、A统计、输入契约与语言/图像缓存文件。
  日志 `outputs/cache_full_validation_20260911/audit_20260912.log`。这不代表逐帧重新编码比较。
- 八卡DeepSpeed小模型保存/恢复通过：`status=PASS, backend=deepspeed, mode=resume, ranks=8`。
  日志 `outputs/cache_full_validation_20260911/tiny_ds8_resume.log`；真实4B保存/恢复仍待执行。
- 修复现有world-policy恢复适配器：从已训练保存点恢复时明确 `initialization=trained`，
  不再继承新预训练的 `flux_backbone` 标记。其余优化器/LR/数据预算不变；
  原始配置不修改。两种初始化来源的回归共4项在本地和190通过，GitHub提交 `b19fb50`。
- 已提交启动上方16条八卡infra测试，输出 `outputs/cache_full_validation_20260911/infra_eight_gpu`，
  总控日志 `infra_eight_gpu.launch.log`；结果尚待验收，不能标记重放或吞吐测试通过。
- 正式120k训练尚未启动；需继续完成重放和真实4B保存/恢复等门槛。

- 全量缓存启动成功，PID1739983；各rank确认为CUDA 0–7，约1.87 GiB/卡。
  2026-09-11 11:40 UTC附近日志已各完成130–140个新episode（打印间隔10个），
  初期单卡约0.205–0.214 episode/s，剩余ETA约4.3–4.5小时；仅为动态估计。
- 更新后的现有保存/恢复诊断已在CPU双进程Gloo执行通过：`status=PASS, mode=resume, ranks=2`。
  对照断点前后online参数、Value EMA、优化器导致的下一步更新、scheduler、RNG与冻结参数均一致。
  日志 `outputs/cache_full_validation_20260911/tiny_gloo_resume.log`。
  这不是八卡DeepSpeed或4B checkpoint验收，后两项仍待执行。
- 自动定时跟进未获权限检查批准，因此没有创建自动化任务，也没有用后台脚本绕过。
  已启动的全量缓存持续运行；剩余验收命令已经准备，但需要本线程继续执行，或用户明确批准跨时段自动跟进后接续。
- 正式120k长训练、正式Round0以及Stage1/Stage2均未启动。
