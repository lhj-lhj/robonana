# 开发与实验交接

当前进度只维护在 [当前实验](MULTITASK_MBRL_PROTOCOL.md)，不要从历史文档的目录名推断训练状态。源码修改前核对 Git、保存配置和实际进程。

## 工作位置

- 仓库：https://github.com/lhj-lhj/robonana ，分支 main。
- 190 checkout：`/data3/hongjia/robonana`；本次 eval 已由用户要求停止。
- 消融验收目录：`/data3/hongjia/robonana_worktrees/world_rope_prefix_20260916`，代码 `f2be651` 已在190通过251项测试。用户随后明确要求合入main：原审阅分支 `76edb7f` 已合入GitHub main并同步190主checkout。71运行中的源码未更新。
- 71 checkout：`/raid/hongjia/robonana`；共享GPU的Round0在容器中，修改源码可能影响新启动子进程，不能无条件同步到正在运行的71。
- 本地 checkout 因机器不同而变化；不要照搬历史的 Windows 路径。
- 190 GPU当前被占用。用户要求先整理文档、准备对照入口；没有授权现在启动新实验。

## 修改和部署

复用 FACT/FLUX 模块；新逻辑放在现有 adapter、mask、loss、配置和测试内。GitHub是源码同步入口：候选改动先提交推送到工作分支，190拉取验证后再合入main并部署。用户要求不在本地跑测试。遇到未提交修改先检查并保留，不能强制重置。

不删除或覆盖数据、缓存、checkpoint和既有结果。只停止明确属于目标实验的进程。代码更新不等于运行进程已加载新版，也不能当作启动实验的授权。

## 当前算法与已确认消融

默认模型 `MacFlux2FACTModel`，动作48步；Stage1训练actor/world，Stage2冻结FLUX训练Q/V。Value独有FP32 EMA。统计文件固定A，缓存/在线输入统一走latents_v2链路。

默认 `build_mac_attention_bias` 的 predicted action 和 clean action 都是双向。用户已确认两组对照及具体实现：`fixed48` 基线，以及 `rope_prefix`（均匀抽h∈[1,48]，不加horizon token，RoPE标记t+h，clean action causal，所有world分支只看动作前h步，reward只监督前h步、success对齐t+h）。只在现有mask函数增加开关；数据、trainer与保存配置使用同一模式。动作chunk仍48，去噪动作仍双向。这条最新指令优先于历史“不允许idx_h”的说明。已准备代码不代表获准启动GPU实验。

旧 `SegmentMap/build_attention_bias` 及其旧forward引用已按用户要求清理；保留MAC实际继承的FLUX模块/helper。旧actor转换测试与删除前保存的CPU输出比较，不再依赖旧mask代码。

吸收态BC已按FACT修复：成功轨迹包含终点帧，终点及其padding使用最终state作绝对hold目标，48步全部监督；失败仍只取完整48步窗口且BC权重为0。不要恢复旧的“成功padding不参与BC”规则。Delta仍减chunk起始state，只有起始state已在终点时关节delta才为0；归一化后为 `-action_mean/action_std`，夹爪保持绝对值。旧120k权重不会随代码修复自动学会停驻，效果要经后续训练和评测验证。

## 验证

用户最新要求：后续验证全部在190进行，不在本地运行测试。允许占GPU跑测试，但不启动正式训练或正式eval。使用190现有环境验证默认mask等价、前缀隔离、目标帧/RoPE一致、配置默认值、旧actor转换和脚本参数；小模型GPU forward/backward选空余显存足够的卡。文档中的计划、单元测试通过、真实实验完成要明确区分。

现行细节见 [技术说明](TECHNICAL_REFERENCE.md)、[代码索引](CURRENT_CODE_MAP.md) 和 [脚本导航](../scripts/README.md)。历史训练、恢复和图像审计保留在archive；不要把某次单任务pilot的默认checkpoint当成50任务实验的初始化。
