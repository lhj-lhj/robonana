# RoboNana

基于 FACT / FLUX.2 的 RoboTwin 动作与 world model，动作 chunk 为 48；后续用 Q/Value 做策略改进。

先看这三个入口：

- [当前实验](docs/MULTITASK_MBRL_PROTOCOL.md)：新120k、Round0进度、成功率排查、两组RoPE/动作前缀消融计划。
- [代码说明](docs/TECHNICAL_REFERENCE.md)：训练、推理、图像管线和模块关系；[代码索引](docs/CURRENT_CODE_MAP.md) 用于找实现。
- [脚本导航](scripts/README.md)：每个入口的作用和是否会启动GPU任务。

## 配置与启动

- [训练配置](configs/train.json)、[评测配置](configs/eval.json)、[续训配置](configs/resume.json)：完整填写后用 `python scripts/run_multitask_mbrl.py <train|eval|resume> --config <文件>` 查看最终计划，`--execute` 才启动。
- GPU、microbatch、累积次数、global batch、步数和学习率缺项或冲突即报错；不再接受旧环境变量或历史实验fallback。详见[配置说明](scripts/README.md)。
- 配置解析、训练组装和续训适配在 `src/robonana/configs/`。新阶段一次生成FACT配置；恢复显式读取来源快照。模型、数据、loss、checkpoint合同保持原有语义。

实验状态请看带时间的实验记录与实际进程，不从仓库默认值推断。代码更新不会自动改变71的运行中eval或190的训练。

## 文档怎么找

`docs/` 放现行说明，`docs/archive/` 放历史实验和排错证据。归档中的“正在运行”“尚未启动”和旧路径只代表当时状态，不是今天的操作指令。入口和路径以当前实验为准。

开发约定见 [AGENTS.md](AGENTS.md) 和 [交接说明](docs/AGENT_HANDOFF.md)。经过测试的源码经 GitHub main 同步；数据、checkpoint、回放和日志不提交，也不覆盖已有实验。
