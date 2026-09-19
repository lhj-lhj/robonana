# 当前实验：50 任务 MBRL

最后整理：2026-09-17。这里记录当前进度、实验参数和下一步；旧的逐次排错日志见 [历史记录](archive/MULTITASK_MBRL_RUN_LOG_20260911_14.md)。

## 当前到哪一步

### 2026-09-20：140k 官方 seed=0 口径的 50×2 全任务评测

用户将协议改为与 RoboTwin 官方 `seed: 0` 完全一致：每个 task/config 从候选 seed 100000 开始，先跑官方 expert check，不可解候选递增跳过，直到 Clean 和 Randomized 分别得到 50 个实际评测 episode；不再使用此前 expert seed cache。共享 GPU 入口现支持 inline expert check，仍保留一张卡一个持久策略服务、低频 scout、仅失败完整回放保存的加速逻辑。聚焦回归在190通过2项，提交 `e39eb4d`。

正式评测已在71容器 `robonana-eval140k-ready-71` 启动，代码 `/raid/hongjia/robonana_eval140k`，输出 `/raid/hongjia/robonana_rollouts/absorbing140k_official_seed0_50_20260920`，主日志 `/raid/hongjia/robonana_deploy/eval140k_official_seed0_50_20260920.log`。使用八张H200、端口9900–9907、140k权重SHA256 `b951e9a3d7e2ce9c2b0717e025247d247eaf8198e1d12c29501f9571b508f095`。低成功率任务排在队首；启动核对为1个主进程、8个策略服务、8个expert worker，八卡均进入计算且未OOM。不要混入旧seed结果或修改冻结的 `protocol.json`。

### 2026-09-19：停止rope_prefix并查看多h预测；准备fixed48 batch256

用户要求停止rope_prefix四卡训练，已只中断 `rn_rope_prefix_120k_4gpu` 并确认该组四rank退出。停止前日志到11650，最新完整保存点为11000；原4–7卡fixed48 batch128继续运行（核对到12020）。

使用rope_prefix step11000生成训练集world拟合诊断：`Clean/blocks_ranking_size` episode0与 `Clean/place_dual_shoes` episode49，各早/中/终点3个窗口，h=1/8/16/32/48，共30组。真实动作条件、20步纯噪声采样、同窗口不同h共享随机噪声；每行当前图/预测t+h/目标t+h均为VAE解码，不是在线策略成功率。修复旧探针默认采样48却读取随机h目标的问题：显式传h，非48自动走完整前向；h48继续用缓存。

图像及指标在190 `/data3/hongjia/robonana/outputs/rope11000_horizon_images_20260919`，本地项目同名outputs目录；可打开index.html。新增采样/配置回归在190独立worktree `horizon_batch256_20260919` 共38项通过，代码c46a612。

用户确认新训练为fixed48，原始FLUX初始化，120k更新，0–3卡，每卡32、累积2、global256、BF16与GC。首轮短测在NCCL初始化失败：系统已将用户态615.71.09库替换进磁盘，但内核仍610.43.02；不是OOM。通过apt下载官方 `libnvidia-compute-610=610.43.02-0ubuntu0.24.04.1` 仅解压到 `/data3/hongjia/opt/nvidia-610-libs/extracted`，新启动脚本设置其用户态库目录至LD_LIBRARY_PATH，未安装系统包或重启驱动。旧fixed48继续使用已加载610库。

新脚本 `/data3/hongjia/run_fixed48_batch256_20260919.sh`，smoke参数执行两步，正式输出 `/data3/hongjia/robonana/experiments/fixed48_batch256_4gpu_20260919/pretrain`；两步短测exit0，峰值allocated40.494GiB/reserved47.908GiB；第二步11.37秒/更新。正式北京时间15:12启动，会话 `rn_fixed48_batch256_120k`，W&B `o50vt88l`，保存配置已确认0–3卡、32×4×2=256、GC、原始FLUX初始化、120k；15:15核对到step10/120000，loss有限，无OOM，global batch256与6075103帧全量数据在日志确认；初始5.09秒/更新、ETA约7天，仅作早期参考。峰值LR保持FLUX2e-5/机器人1e-4，warmup500、seed6666，每1000步保存。


### 2026-09-18：四卡双组120k消融已启动

用户确认从同一原始FLUX初始化，两组各训练120000步。启动代码 `885b49a`，独立目录 `/data3/hongjia/robonana_worktrees/rope_dense_reward_20260918`，包含dense Reward解耦修复；190主checkout未替换。

| 模式 | GPU | tmux | W&B |
|---|---|---|---|
| rope_prefix | 0–3 | rn_rope_prefix_120k_4gpu | fy3rcid9 |
| fixed48 | 4–7 | rn_fixed48_120k_4gpu | um4fhyb2 |

共同参数：每卡16、累积2、global batch128、BF16、gradient checkpointing、seed6666；原始FLUX骨干加相同seed初始化的机器人头，全新optimizer。峰值LR为FLUX 2e-5/机器人1e-4，warmup500、120k衰减；成功演示Clean+Randomized、A统计、成功吸收态修复；每1000步保存。唯一算法差异为已确认的world-conditioning两组设定。

输出根目录 `/data3/hongjia/robonana/experiments/world_ablation_4gpu_20260918/{rope_prefix,fixed48}/pretrain`。启动脚本 `/data3/hongjia/launch_world_ablation_4gpu_20260918.sh <mode>`；已有输出不能直接重跑覆盖。正式启动北京时间18:29，18:31两组均确认4个rank和batch128。18:32两组均到step10，loss有限、reward_valid_fraction=1；rope_prefix约6.72秒/更新，fixed48约6.43秒/更新，初始ETA约9天（仅10步，未计稳定后变化及保存开销）。GPU尚余约56–59GiB。后续进度以日志为准。

全量缓存核验：27500 episodes、6075103帧全部通过。四卡配置14项测试通过；rope_prefix四卡两步显存短测通过。启动时各卡已有约82GiB其他任务占用，GPU算力共享，耗时不能套用独占八卡速度。未设置定时。


### 最新：140k缓存seed预评测（2026-09-17）

用户确认合并71本地的 `51a18fb`（此前没推GitHub）；已保留作者合入main，并补充expert异常日志。新评测代码 `cac490d` 已同步190，缓存seed/八卡分片/失败保存共23项回归在190通过。71为独立checkout，不修改正在采集seed的原目录。

140k在13:36 UTC完成全部更新，13:37保存权重；进程清理时发生SIGSEGV。已验证完整权重可加载、392个tensor/5,002,609,664参数、step140000且fixed48，推理服务也已成功加载。权重SHA256为 `b951e9a3d7e2ce9c2b0717e025247d247eaf8198e1d12c29501f9571b508f095`，71传输后哈希一致。

seed实际情况与用户提供的概述不同：71 `expert_seed_cache` 的一些manifest只有新增部分，双鞋目录尚未建立。用户明确允许合并旧Round0已验收seed+instruction，不重新采集。冻结清单合并71新manifest、71旧accepted jobs、190旧ledger及已有prepare/accepted_seeds.json，按(task,config,seed)去重并保留manifest_source。不能重复seed凑100，不能根据模型成败挑seed。

| 预检task | Clean实际seed | Randomized实际seed | 旧新120k成功率（已完成样本） |
|---|---:|---:|---|
| blocks_ranking_size | 92 | 98 | 12/57；2/11 |
| place_dual_shoes | 100 | 100 | 22/100；31/100 |

旧actor的Clean参考分别44/50和用户报告约86%；不同seed/样本量不能直接当严格配对。汇总新成绩时同时统计与旧120k重合的seed。缺少的积木8+2条需从后续完成的缓存或其他已有验收记录补齐；禁止启动新expert采集。

- 190预检：`/data3/hongjia/robonana_rollouts/absorbing140k_probe_20260917_r2`，启动日志同路径加 `.launch.log`；8个共享GPU通道，端口8700–8707，action_only、scout_replay，只发布校验通过的失败数据。
- 冻结seed：190 `/data3/hongjia/expert_seed_cache_140k_20260917`，71 `/raid/hongjia/expert_seed_cache_140k_20260917`。预检协议文件保存实际jobs与分片；不要在运行中修改清单或protocol。
- 190系统nvidia-smi缺失但CUDA正常，已将71真实查询工具复制到 `/data3/hongjia/opt/nvidia-tools/nvidia-smi`；启动时把此目录加入PATH。未改系统驱动。首轮 `absorbing140k_probe_20260917` 无有效结果，保留排错。
- 71：代码 `/raid/hongjia/robonana_eval140k`，权重/配置 `/raid/hongjia/robonana_deploy/absorbing140k/{transformer/diffusion_pytorch_model.bin,config.json}`，容器 `robonana-eval140k-ready-71` 已就绪（尚未启动全量评测）。镜像 `robonana-eval:20260915`，必须加 `--runtime=nvidia`；映射 `/raid/hongjia` 到容器的同路径和 `/data3/hongjia`，保留原图像/归一化路径。

后续已获授权：预检完成后判断两项任务是否恢复，并确认无渲染/回放异常；正常则启动两机全量，先跑齐100条的task/config。使用 `collect --expert-seed-cache ... --shared-gpus --ready-only`（collect代表action-only及失败采集，eval子命令是Stage2 Q筛选，不能混用）。两机各8通道时 `--shard-count 16`，190 offset0、71 offset8，配同一冻结jobs，避免重复。已预检两task应复用结果，剩余任务单独启动；后续新齐的配置使用新wave输出，不能改已运行协议。全量不能用 `--allow-partial-expert-seeds`。

预检尚未完成；早期完成样本有“成功快、失败回放慢”的偏差，不要把早期完成SR当最终结果。全量尚未启动。

预训练已完成：从原八卡 119000 保存点续到 120000。Round0 原本在 190 和 71 各跑 25 个任务，190 已按用户要求停止，71 上次检查仍在运行。Stage1、Stage2 尚未启动。

2026-09-17 16:04（北京时间）已在190启动吸收态修复后的fixed48续训，120k→140k，八卡总batch128；16:07核对已更新到120060，详见下面的追加适配记录。

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

已确认并实现：每样本均匀抽 h∈[1,48]，image/future-state/success 对齐 t+h，reward 独立监督完整48步（包括成功吸收尾段），RoPE时间为0。U/S'/I'受前缀限制且禁止读取R，避免完整chunk动作经reward间接泄漏；clean action仍causal。动作 BC 仍监督完整有效 48 步。成功尾段继续沿用吸收态规则，失败尾段不补造 transition。数据先检查完整48步窗口，再选择world目标，不因h变短放过缺失transition。

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

开关写入 `models.world_conditioning` 和各训练数据池；保存后由checkpoint配置恢复。旧配置缺少该字段时按 `fixed48` 读取，推理不会自动换语义。新分支的图像RoPE使用现有图像时间轴，success/future-state使用现有robot时间轴，reward时间固定0；没有新增token或参数。动作only推理不受h影响；world缓存推理目前仍查询h=48，和完整forward保持一致。此消融入口仅支持pretrain/Stage1，Stage2新协议不在本次范围。

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

续训恢复专项回归已在190通过24项，完整CPU回归260项通过、4项CUDA测试跳过。测试包含原FACT曲线峰值、终点、Adam不变以及中途保存恢复；本地未跑测试。日志：190 `/tmp/robonana_fixed48_20k_20260917_pytest.log`。

实际启动与核对：

- 启动代码 `78101cd`（核心续训适配 `963f5bc`），190 main；tmux会话 `rn_absorbing_fixed48_20k_20260917`。
- 启动时间2026-09-17 08:04:44 UTC（北京时间16:04:44），正式日志 `<输出目录>/logs/train_20260917T080444Z.log`。
- [W&B运行 a5tjqsuk](https://wandb.ai/hongjia-liu-aalto-university/robonana/runs/a5tjqsuk)。
- 日志确认原120k的DeepSpeed模型、优化器、scheduler和自定义进度恢复成功；`WORLD CONTINUATION VERIFIED` 显示step120000、max_steps140000、scheduler_start120000。最初LR为原FACT warmup首点：FLUX `2e-5/501`，机器人 `1e-4/501`。
- 八个训练rank、总batch128；16:07核对已到120060，action_loss约0.0093、total_loss约0.4843，BC样本比例1.0，无非有限值错误。实际约0.93秒/步，初步预计5–6小时完成新增20k（保存开销和后续速度可能改变）。
- 原120k权重SHA256重新校验一致；原数据/优化器/八卡配置逐项比对一致。此记录是启动快照，不代表训练已经完成或成功率恢复。

| 阶段 | 数据/初始化 | 更新步数 | 有效 batch |
|---|---|---:|---:|
| Pretrain（已完成） | 原始 FLUX；50任务 Clean+Randomized，27500示范 | 120000 | 8×16×1=128 |
| Round0（部分完成） | 新120k；每任务 Clean100+Randomized100 | 无 | 评测 |
| Stage1（未开始） | 新120k；原成功示范50%＋Round0失败50% | 60000 | 128 |
| Stage2（未开始） | Stage1；冻结FLUX，训练Q/V | 20000 | 128 |
| Final eval（未开始） | Stage2；Round0锁定场景，32候选Q选择 | 无 | 评测 |

FLUX/机器人 LR：Pretrain 2e-5/1e-4，Stage1 2e-5/2e-5，Stage2 冻结/1e-4。各阶段新建优化器时钟，warmup500。VAE FP32，FLUX BF16，Value EMA FP32；统一 A 统计与 latents_v2 输入缓存。

常用计划入口：`python scripts/run_multitask_mbrl.py train --phase pretrain --output <新目录>`。默认只打印计划，`--execute` 才启动。`audit` 会读取实际数据和缓存做 CPU 检查。完整旧命令与验收记录在历史文档中；其中撤回的四卡续训方案不是当前操作步骤。

### 2026-09-18 rope_prefix 解耦验收

R读取完整G，U/S'/I'禁止读取R并只读取G前h步；R的训练与缓存RoPE时间均为0。reward标签由chunk_delta生成，与h独立，成功吸收后缀全部监督。success的>=写法与原先clipped future_index判定等价，原实现并未漏掉越过终点的正标签。fixed48保留原有拓扑及数值语义。

190独立worktree `rope_dense_reward_20260918`：专项16项通过（原14项加两项边界/固定模式回归）；数据合同、loss、world/prefix缓存和CPU/CUDA BF16反传等相关回归合计51项全部通过（14.47秒，无跳过）。主checkout评测进程未更新，未启动训练。
