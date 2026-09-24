# 开发与实验交接

当前进度只维护在 [当前实验](MULTITASK_MBRL_PROTOCOL.md)，不要从历史文档的目录名推断训练状态。源码修改前核对 Git、保存配置和实际进程。

## 工作位置

- 2026-09-19更新：rope_prefix四卡已按用户要求停止，最新ckpt11000；多h图像诊断30组已完成。原fixed48 batch128仍运行，新fixed48 batch256（0–3卡，每卡32×累积2，GC）短测通过后启动。新任务使用隔离worktree `horizon_batch256_20260919` 和用户目录610驱动库；详见当前实验最新记录，不要重启系统驱动。

- 2026-09-18新增：用户明确授权并已启动原始FLUX双组120k消融，rope_prefix卡0–3、fixed48卡4–7，每卡16×累积2、global128、GC开启。代码885b49a位于独立worktree `rope_dense_reward_20260918`；会话/输出见当前实验。不得把此前“未授权训练”的历史记录用于否定此次授权。

- 仓库：https://github.com/lhj-lhj/robonana ，分支 main。
- 190 checkout：`/data3/hongjia/robonana`；本次 eval 已由用户要求停止。
- 消融验收目录：`/data3/hongjia/robonana_worktrees/world_rope_prefix_20260916`，代码 `f2be651` 已在190通过251项测试。用户随后明确要求合入main：原审阅分支 `76edb7f` 已合入GitHub main并同步190主checkout。71运行中的源码未更新。
- 71 checkout：`/raid/hongjia/robonana`；共享GPU的Round0在容器中，修改源码可能影响新启动子进程，不能无条件同步到正在运行的71。
- 本地 checkout 因机器不同而变化；不要照搬历史的 Windows 路径。
- 2026-09-17用户明确授权的190八卡120k→140k续训已启动：吸收态修复、fixed48及原batch128。保留Adam状态，原峰值LR与FACT WarmupCosine模块，新增20k重新走500步warmup。启动代码78101cd，tmux `rn_absorbing_fixed48_20k_20260917`，W&B `a5tjqsuk`；16:07核对更新到120060，实时状态需查日志。不要停止或改动这次训练，具体路径见当前实验。
- 最新：140k已保存且验证可加载；用户授权先八卡评测blocks_ranking_size和place_dual_shoes，恢复正常后自动起190+71全量评测（每task Clean100/Randomized100、只保存失败）。预检在190的 `absorbing140k_probe_20260917_r2` 输出运行，tmux `rn_eval140k_probe`；首轮输出因缺少nvidia-smi而启动失败，不能算模型失败。详见当前实验。71原seed采集checkout不动，评测使用 `/raid/hongjia/robonana_eval140k`。

## 当前启动方式（2026-09-20）

新配置只读完整JSON，入口 `scripts/run_multitask_mbrl.py train|eval|resume --config ...`；无旧环境变量兼容，缺字段/类型错误/batch冲突立即报错。见[脚本配置说明](../scripts/README.md)。7个历史配置模块和单任务轮次入口已删除，旧实验记录的命令只作历史证据。

本次重构在独立分支/worktree验证；不要把它热更新到190正在训练或71正在eval的checkout。运行进程继续使用原有快照。

## 修改和部署

复用 FACT/FLUX 模块；新逻辑放在现有 adapter、mask、loss、配置和测试内。GitHub是源码同步入口：候选改动先提交推送到工作分支，190拉取验证后再合入main并部署。用户要求不在本地跑测试。遇到未提交修改先检查并保留，不能强制重置。

不删除或覆盖数据、缓存、checkpoint和既有结果。只停止明确属于目标实验的进程。代码更新不等于运行进程已加载新版，也不能当作启动实验的授权。

## 当前算法与已确认消融

默认模型 `MacFlux2FACTModel`，动作48步；Stage1训练actor/world，Stage2冻结FLUX训练Q/V。Value独有FP32 EMA。统计文件固定A，缓存/在线输入统一走latents_v2链路。

默认 `build_mac_attention_bias` 的 predicted action 和 clean action 都是双向。用户已确认两组对照及具体实现：`fixed48` 基线，以及 `rope_prefix`（均匀抽h∈[1,48]，不加horizon token，RoPE标记t+h，clean action causal，U/S'/I'只看动作前h步且不读取R；R独立读取完整48步动作并监督完整chunk，RoPE时间为0；success对齐t+h）。只在现有mask函数增加开关；数据、trainer与保存配置使用同一模式。动作chunk仍48，去噪动作仍双向。这条最新指令优先于历史“不允许idx_h”的说明。已准备代码不代表获准启动GPU实验。

旧 `SegmentMap/build_attention_bias` 及其旧forward引用已按用户要求清理；保留MAC实际继承的FLUX模块/helper。旧actor转换测试与删除前保存的CPU输出比较，不再依赖旧mask代码。

吸收态BC已按FACT修复：成功轨迹包含终点帧，终点及其padding使用最终state作绝对hold目标，48步全部监督；失败仍只取完整48步窗口且BC权重为0。不要恢复旧的“成功padding不参与BC”规则。Delta仍减chunk起始state，只有起始state已在终点时关节delta才为0；归一化后为 `-action_mean/action_std`，夹爪保持绝对值。旧120k权重不会随代码修复自动学会停驻，效果要经后续训练和评测验证。

吸收态修复 `2efd4f4` / 测试夹具修正 `de4ee4c` 已在190通过全部261项回归（257项CPU加4项CUDA），并用鞋子/瓶子真实演示、现有缓存与A统计验证终点目标及48步BC反传。详见当前实验的验收记录；没有启动正式训练或eval。

## 验证

后续验证全部在190进行，不在本地运行测试。2026-09-17已授权上述fixed48吸收态20k正式续训；这不代表授权启动其他消融或正式eval。文档中的计划、单元测试通过、真实实验完成要明确区分。

现行细节见 [技术说明](TECHNICAL_REFERENCE.md)、[代码索引](CURRENT_CODE_MAP.md) 和 [脚本导航](../scripts/README.md)。历史训练、恢复和图像审计保留在archive；不要把某次单任务pilot的默认checkpoint当成50任务实验的初始化。

## V3 Action expert（2026-09-24）

用户2026-09-24已进一步授权停WAM并正式训练。分支 `codex/action-expert-v3-20260924`：
独立 flow Action expert、联合训练保留 C K/V 梯度、现有 robot LR、fixed48 + 吸收态。
配置模板 `configs/train_v3.json`（global256、120k），复用统一入口；用户本地注释已保留提交。
190 隔离 worktree `action_expert_v3_20260924` 完整回归293通过，追加专项13通过；
未更新运行实验的 checkout，未合入 main。实现与验证边界见 `INHERITANCE.md` 的 V3 节。

2026-09-24正式启动：v3 fixed48，原始FLUX初始化120k，8×32×1=global256，GC stride1。
真实八卡两步短测exit0；会话`rn_v3_fixed48_bs256_20260924`，实际配置与输出见当前实验顶部。
不要把模板的8×16×2或此前“未授权训练”作为本次运行配置/授权状态。

2026-09-24后续：用户要求停止上面旧任务、pretrain移除Q/V并测试stride2。
代码`5fad5ed`已在190通过301项回归；Q/V仅Stage2创建，GC日志和配置统一为model-native。
8×32×1的stride2在GPU4–7真实OOM，保留stride1；新会话`rn_v3_no_critics_20260924`，
新输出`v3_no_critics_stride1_20260924`。旧任务410步尚无checkpoint，故从原始FLUX重启。
实际配置、失败短测证据和路径见当前实验顶部。
