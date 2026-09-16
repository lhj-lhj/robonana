# RoboNana

基于 FACT / FLUX.2 的 RoboTwin 动作与 world model，动作 chunk 为 48；后续用 Q/Value 做策略改进。

先看这三个入口：

- [当前实验](docs/MULTITASK_MBRL_PROTOCOL.md)：新120k、Round0进度、成功率排查、两组RoPE/动作前缀消融计划。
- [代码说明](docs/TECHNICAL_REFERENCE.md)：训练、推理、图像管线和模块关系；[代码索引](docs/CURRENT_CODE_MAP.md) 用于找实现。
- [脚本导航](scripts/README.md)：每个入口的作用和是否会启动GPU任务。

## 当前状态

2026-09-16 最近核对：新预训练120k完成；190上的Round0已停止，71上次检查仍在跑。Stage1/Stage2未开始。实验文档中的数字是带上下文的快照，不是实时监控。

当前默认 MAC 的 clean action 是双向注意力；已加入用户确认的 `rope_prefix` 消融：clean action causal、world 只读前 h 步、RoPE 标记目标帧。代码已准备，尚未启动实验；当前120k仍属于默认基线。

旧120k原始格式需显式转换；已有 actor-preserving 导出可用于 action-only 复测。导出不代表旧模型采用了现在的训练图像管线。不要混用旧、新120k的模型、统计口径或运行结果。

## 文档怎么找

`docs/` 放现行说明，`docs/archive/` 放历史实验和排错证据。归档中的“正在运行”“尚未启动”和旧路径只代表当时状态，不是今天的操作指令。入口和路径以当前实验为准。

开发约定见 [AGENTS.md](AGENTS.md) 和 [交接说明](docs/AGENT_HANDOFF.md)。经过测试的源码经 GitHub main 同步；数据、checkpoint、回放和日志不提交，也不覆盖已有实验。
