# 当前实验：50 任务 MBRL

最后整理：2026-09-17。这里记录当前进度、实验参数和下一步；旧的逐次排错日志见 [历史记录](archive/MULTITASK_MBRL_RUN_LOG_20260911_14.md)。

## 当前到哪一步

预训练已完成：从原八卡 119000 保存点续到 120000。Round0 原本在 190 和 71 各跑 25 个任务，190 已按用户要求停止，71 上次检查仍在运行。Stage1、Stage2 尚未启动。

| 分片 | 最近核对的有效 episode | 成功 | 状态 |
|---|---:|---:|---|
| 190_a | 615 | 394 | 已停止 |
| 190_b | 319 | 162 | 已停止 |
| 71_a | 1022 | 835 | 上次检查运行中 |
| 71_b | 153 | 106 | 上次检查运行中 |

这是停止 190 时的快照，不是实时进度。合计 2109/10000，成功 1497/2109（71.0%）。未评满的任务不能当作完整 50 任务成绩；专家检查拒绝、超时和渲染错误不算模型失败。

模型文件（190）：

```text
/data3/hongjia/robonana/experiments/multitask_mbrl_v1/pretrain119k_8gpu/
  config.json
  models/checkpoint_epoch_3_step_120000/transformer/diffusion_pytorch_model.bin
```

权重 SHA256：`cd0987b5061221185fd39be783e64c3cdeab6fd46dff31aca27f60ef046c200e`。

| 机器 | 输出 |
|---|---|
| 190 | `/data3/hongjia/robonana_rollouts/multitask_round0_v1`、`multitask_round0_v1_190_b` |
| 71 | `/raid/hongjia/robonana_rollouts/multitask_round0_v1_71_a`、`multitask_round0_v1_71_b` |

任务分配在 190 的 `outputs/round0_71_deploy_20260915/task_allocation.json`，71 的 `/raid/hongjia/robonana_deploy/task_allocation.json`。原有完成结果保留；恢复前要检查并隔离未提交的 attempt，不能覆盖。

## 已完成的诊断

190 后半程约 15.1 小时只新增 258 条，71 同期新增 860 条。190 进入低成功率、长轨迹任务，失败需要逐帧回放；`open_laptop` 还有 488 次准备失败、零有效结果。ledger 的 `candidate_rejected` 标签本身不能区分正常专家拒绝与准备进程异常，仍需看 prepare.log。

190/71 每条有效 episode 平均 scout 约 104/100 秒；失败回放约 417/422 秒。190 失败比例约 40.5%，71 约 19.8%，所以平均 scout+replay 耗时为 272/184 秒。任务静态分片失衡是主要已知问题；共享 GPU 的影响没有独占对照，不能从 100% 利用率直接量化。此前把短窗口外推成稳定 ETA 不可靠。

旧 120k 的 91.7% 来自 46 个完成任务 × Clean 50（2109/2300），不包含 4 个 ERROR 任务。本次是新的训练权重、不同 seed，并含 Randomized。但只看 Clean，部分同名任务仍有大差距：

| task | 历史旧 120k Clean | 当前新 120k Clean |
|---|---:|---:|
| adjust_bottle | 50/50 | 100/100 |
| beat_block_hammer | 47/50 | 66/100 |
| blocks_ranking_rgb | 50/50 | 26/89 |
| blocks_ranking_size | 44/50 | 12/57 |
| hanging_mug | 20/50 | 36/100 |

这些结果证明有差异，尚未确定差异由训练、seed、输入或渲染中的哪一项造成。权重文件大小和加载参数总数不能单独解释 actor 能力，尤其旧 actor 导出包含新初始化且 action-only 不使用的 world/critic 参数。

## 下一步 A：旧 actor 在当前评测管线中复测

状态：**仅准备，未启动。190 GPU 被占用，等待用户另行安排运行。**

先测 `blocks_ranking_rgb`、`blocks_ranking_size`、`beat_block_hammer` 的 Clean；每任务 50 条。加载已经导出的旧 actor：

```text
checkpoints/120k_action_only_export_20260909/diffusion_pytorch_model.bin
checkpoints/120k_action_only_export_20260909/model_config.json
```

它保留旧 actor 的权重映射，运行时只允许 action-only。旧导出 SHA256 为 `c248bc56e424bcd070584e9db5474b405fbf156284cf7144274372285d12758c`；这不是 8 月历史原始权重文件的哈希，不能混称。当前图像处理、渲染器、A 统计、采样参数需随结果记录。

现有入口：[eval_robotwin_all_tasks_parallel.sh](../scripts/eval_robotwin_all_tasks_parallel.sh)。下面是将来使用的实际启动命令，**执行就会占 GPU，本次没有执行**；4/5 只是预留示例，启动前按空闲卡调整。

```bash
cd /data3/hongjia/robonana
ROBONANA_TRAINED_CHECKPOINT="$PWD/checkpoints/120k_action_only_export_20260909/diffusion_pytorch_model.bin" \
ROBONANA_MODEL_CONFIG="$PWD/checkpoints/120k_action_only_export_20260909/model_config.json" \
ROBONANA_INFERENCE_MODE=action_only \
ROBONANA_EVAL_TASKS=blocks_ranking_rgb,blocks_ranking_size,beat_block_hammer \
ROBONANA_EVAL_SERVER_GPUS=4 ROBONANA_EVAL_SIM_GPUS=5 \
ROBONANA_EVAL_JOBS_PER_GPU=1 ROBONANA_EVAL_SEED_GROUP=0 \
ROBONANA_EVAL_RUN_DIR="$PWD/outputs/old120k_current_pipeline_clean50" \
EVAL_VIDEO_LOG=0 bash scripts/eval_robotwin_all_tasks_parallel.sh demo_clean 50
```

该入口使用当前专家检查生成可运行场景。即便候选 seed 起点与历史相同，最终 accepted seeds、指令和渲染像素也未必相同，必须核对输出。不要直接把历史日志打印的 `current seed` 当作本条实际 seed，先查对应历史评测代码的递增时机。

更严格的第二步：锁定同一批 seed 和完整 instruction，用当前管线分别跑旧、新 actor，只用 scout，不做失败回放。复用 [benchmark_robotwin_collection_pool.py](../scripts/diagnostics/benchmark_robotwin_collection_pool.py) 的 `--jobs-json --inference-mode action_only --capture-mode scout`。同一清单用于两模型，报告覆盖率与逐 seed 结果，错误不静默换种子。该清单尚未生成。

若旧 actor 在当前管线也明显下降，说明应优先排查管线变化，但不能只凭该结果断言是渲染器；预处理、指令、数值精度和采样也需要逐项核对。若旧 actor 恢复高成功率而新 actor 低，则更支持训练/模型差异这一方向。

## 下一步 B：两组 world 训练对照

状态：**用户已确认实现方案；代码和启动入口已准备，未启动训练。**

用户已明确只比较两组，不增加第三组：

| 项目 | A：当前基线 | B：目标帧 RoPE + 动作前缀 |
|---|---|---|
| 每次输出动作 | 48 步，去噪分支双向 | 同左 |
| world 目标 | t+48 | t+h |
| clean action 自注意力 | 双向 | causal，动作 j 只看 ≤j |
| world 可见动作 | 完整 48 步 | 仅 a[t:t+h]，不含 a[t+h] |
| horizon 表达 | 当前位置编码 | 用现有 RoPE 标记目标帧 h |
| 额外 horizon token/embedding | 无 | 无 |

这里的 h 是训练样本里的整数目标帧偏移，不是新增的模型 token。输入、目标和 RoPE 必须对应同一个 h；只修改位置编码而仍监督 t+48 不构成这个实验。

已确认并实现：每样本均匀抽 h∈[1,48]，image/future-state/success 对齐 t+h，reward 只监督前 h 步。所有 world 分支都受前缀限制，避免信息经 reward/success 或 clean action 间接泄漏。动作 BC 仍监督完整有效 48 步。成功尾段继续沿用吸收态规则，失败尾段不补造 transition。数据先检查完整48步窗口，再选择world目标，不因h变短放过缺失transition。

消融代码只在现有 `build_mac_attention_bias` 加开关，沿用 `MacSegmentMap` 和当前 FLUX wrapper。旧 `build_attention_bias`、`SegmentMap` 及仅服务于旧 forward 的依赖已清理，不再维护两套 mask。

当前训练路径的双向实现可在 `models/attention_mask.py::build_mac_attention_bias` 看到；删除前的旧 causal 实现在 `build_attention_bias` 中，但正式 MAC 没有调用它。消融是对用户指定两项变化的联合比较，不能单独归因于 h 或 causal。

两组使用相同初始化、数据、随机种子、optimizer/LR、有效 batch 和更新步数。旧实验 batch=256（用户说明为128×累积2），当前=128；暂不把 batch 作为第一优先原因，但120k步对应的样本总暴露量不同，不能仅以步数相同认定充分训练等价。

入口复用 [run_multitask_mbrl.py](../scripts/run_multitask_mbrl.py) 的 `train --world-conditioning`。以下两条**只打印计划，不启动训练**；将来明确安排运行时才加 `--execute`。输出使用独立新目录，不能覆盖现有120k。

```bash
python scripts/run_multitask_mbrl.py train --phase pretrain \
  --world-conditioning fixed48 --output experiments/world_ablation/fixed48
python scripts/run_multitask_mbrl.py train --phase pretrain \
  --world-conditioning rope_prefix --output experiments/world_ablation/rope_prefix
```

两组均从原始FLUX初始化，seed=6666、有效batch128、120k更新；只比较上述联合改动。若改成从同一120k继续适配，使用 `--phase stage1 --checkpoint <同一权重> --model-config <同一配置> --replay-root <同一失败池>`，两组都运行相同60k预算；不要把只追加训练的新分支直接与未追加的旧120k当作严格对照。

开关写入 `models.world_conditioning` 和各训练数据池；保存后由checkpoint配置恢复。旧配置缺少该字段时按 `fixed48` 读取，推理不会自动换语义。新分支的图像RoPE使用现有图像时间轴，reward/success/future-state使用现有robot时间轴；没有新增token或参数。动作only推理不受h影响；world缓存推理目前仍查询h=48，和完整forward保持一致。此消融入口仅支持pretrain/Stage1，Stage2新协议不在本次范围。

CPU验证覆盖mask、跨层梯度泄漏、目标帧与RoPE、成功吸收尾段、失败完整窗口、默认行为、旧actor转换、保存配置和BF16训练backward。真实GPU训练和RoboTwin评测等待卡空闲后再运行；CPU通过不等于训练消融已有结论。

### 2026-09-16 验收

用户指定后续测试全部在190进行，并允许GPU小测试，暂不启动正式训练。代码提交 `f2be651` 在190的独立目录 `/data3/hongjia/robonana_worktrees/world_rope_prefix_20260916` 验收：

- 完整pytest：247项通过，含两组CPU/GPU BF16 forward/backward、world缓存和全部新消融测试。
- 原有4项CPU/Gloo双进程测试因只暴露一张GPU触发 `invalid device ordinal`；用 `CUDA_VISIBLE_DEVICES= ACCELERATE_USE_CPU=true` 按CPU模式重跑，4项全部通过。合计251项通过，无代码测试遗留失败。
- Python编译、提交diff格式检查通过。两组 `train --world-conditioning ...` dry-run入口实际执行，只打印配置，未创建实验目录。
- 日志：190的 `/tmp/robonana_world_rope_prefix_20260916_pytest.log` 和 `/tmp/robonana_world_rope_prefix_20260916_distributed.log`。

上述GPU验证使用小模型，不是4B正式训练或成功率评测。用户随后要求先合入main再修复吸收态BC：原审阅分支 `76edb7f` 已合入GitHub main并同步190主checkout。71运行中的源码未更新，后续验证仍全部在190进行。

## 成功尾部吸收态 BC 修复

用户报告 `place_dual_shoes` 放到位后仍移动、破坏稳定成功判定，并指出86%→27%的下降。已确认代码存在成功尾部监督缺口：action clip到倒数第二帧、padding的BC mask为0，而且成功采样排除了最终观察帧。这个缺口需要修复；是否解释全部成功率差异仍需训练后复测。

对齐190的FACT `_get_query_indices` 和完整chunk Action MSE：

- 成功样本包含最后一帧；action索引clip到 `length-1`，终点和超长部分使用最终state作为绝对hold目标，48步 `action_valid_mask` 全为True。
- 原始LeRobot末行action已等于最终state（抽查 `place_dual_shoes` / `adjust_bottle` 的episode0确认）；HDF5回放末行只是上一条command的占位，不能直接把它当终点保持目标。
- 所有action仍减chunk起始state的12个关节维度；夹爪两维保持最终绝对值。到终点前，padding是 `q_final-q_current`；在终点才是raw delta=0，归一化值为 `-action_mean/action_std`。
- 失败窗口范围、transition有效性检查、无尾部补造规则、BC权重0均保持。reward/success的吸收态规则也保持。
- 两组world消融共用这次BC修复；不能再把修复前120k与仅追加训练的新模型当成严格的两组对照。

源码修复不会改变已有120k权重的行为；后续训练与正式成功率复测尚未启动。回归覆盖真实HDF5/LeRobot适配器读取、FACT末尾索引、终点零位移、夹爪保持、非零统计归一化、尾部BC梯度及失败轨迹不变。

### 190 验收结果（2026-09-16）

修复代码 `2efd4f4`、测试夹具修正 `de4ee4c`，在 `/data3/hongjia/robonana_worktrees/absorbing_state_fact_20260916` 验证，本地未跑测试：

- 完整CPU回归：257项通过、4项CUDA测试跳过；再在GPU0补跑这4项，全部通过。合计261项测试均通过。编译和diff格式检查通过。
- 实际读取 `Clean/place_dual_shoes` 和 `Clean/adjust_bottle` 的episode0、现有图像/语言缓存及A统计，检查倒数第三帧与终点帧：48步BC mask全部激活，48步均有非零反传梯度，尾部还原为最终姿态。
- 两个终点样本的12维关节raw delta最大绝对误差均约 `1.86e-9`，夹爪保持最终绝对值。瓶子倒数第三帧的delta约 `0.02236`，符合相对当前state到最终姿态的目标，未错误置零。
- 完整回归日志：190 `/tmp/robonana_absorbing_fact_20260916_pytest.log`。现有Pillow弃用警告不影响通过结果。

未启动正式训练或RoboTwin eval，未改71运行中的源码；这些结果验证训练目标和代码回归，不代表成功率已恢复。

## 原定完整实验参数

### 追加吸收态适配：120k → 140k（2026-09-17）

用户已授权在190八卡续训20k；这是原成功演示上的吸收态修复适配，不是混入失败回放的Stage1，也不启用rope_prefix。

- 来源：上文 `pretrain119k_8gpu` 的完整120000 checkpoint，恢复权重、八卡ZeRO Adam、随机状态及全局步数。
- 输出：`experiments/multitask_mbrl_v1/absorbing_fixed48_120k_to140k_8gpu_20260917`，原120k不改。
- 原参数保持：fixed48、每卡16×8×累积1=128、BF16、无梯度重计算、原Clean+Randomized成功演示和A统计、原loss权重、每1000步保存。
- 原scheduler保存的LR为0，不能直接原样续到140k。用户确认保持原峰值和LR模块：FLUX `2e-5`、机器人 `1e-4`，复用FACT WarmupCosine；新增20k以本地步数0..20000运行、warmup500，Adam和全局步数不重置。曲线起点120000写入配置，后续中断恢复不重新warmup。
- 入口：`world_policy_resume.config` 增加可选 `ROBONANA_ADDITIONAL_STEPS=20000`，默认0仍是原样恢复。

```bash
cd /data3/hongjia/robonana
export ROBONANA_PYTHON=/data3/hongjia/conda/envs/robonana/bin/python
export ROBONANA_RESUME_CONFIG="$PWD/experiments/multitask_mbrl_v1/pretrain119k_8gpu/config.json"
export ROBONANA_RESUME_CHECKPOINT="$PWD/experiments/multitask_mbrl_v1/pretrain119k_8gpu/models/checkpoint_epoch_3_step_120000"
export ROBONANA_PROJECT_DIR="$PWD/experiments/multitask_mbrl_v1/absorbing_fixed48_120k_to140k_8gpu_20260917"
export ROBONANA_ADDITIONAL_STEPS=20000
export ROBONANA_GRADIENT_CHECKPOINTING=0
NCCL_NVLS_ENABLE=0 bash scripts/run_robotwin_train.sh --config robonana.configs.world_policy_resume.config
```

续训恢复专项回归已在190通过24项，包含原FACT曲线峰值、终点、Adam不变以及中途保存恢复。启动状态待核对实际日志，不能以此命令存在判断已经运行。

| 阶段 | 数据/初始化 | 更新步数 | 有效 batch |
|---|---|---:|---:|
| Pretrain（已完成） | 原始 FLUX；50任务 Clean+Randomized，27500示范 | 120000 | 8×16×1=128 |
| Round0（部分完成） | 新120k；每任务 Clean100+Randomized100 | 无 | 评测 |
| Stage1（未开始） | 新120k；原成功示范50%＋Round0失败50% | 60000 | 128 |
| Stage2（未开始） | Stage1；冻结FLUX，训练Q/V | 20000 | 128 |
| Final eval（未开始） | Stage2；Round0锁定场景，32候选Q选择 | 无 | 评测 |

FLUX/机器人 LR：Pretrain 2e-5/1e-4，Stage1 2e-5/2e-5，Stage2 冻结/1e-4。各阶段新建优化器时钟，warmup500。VAE FP32，FLUX BF16，Value EMA FP32；统一 A 统计与 latents_v2 输入缓存。

常用计划入口：`python scripts/run_multitask_mbrl.py train --phase pretrain --output <新目录>`。默认只打印计划，`--execute` 才启动。`audit` 会读取实际数据和缓存做 CPU 检查。完整旧命令与验收记录在历史文档中；其中撤回的四卡续训方案不是当前操作步骤。
