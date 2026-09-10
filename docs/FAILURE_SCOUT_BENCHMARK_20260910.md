# 120k actor：失败轨迹 scout/replay 测试

本次使用固定48帧的120k action-only导出权重，GPU6做推理、GPU7做仿真。
先暂停独立学生蒸馏；最新完整保存点为step2000，暂停前约step2710。

## 四个历史ERROR任务

初测目录（190）：`outputs/eval4_missing_recheck_20260910`。
每任务2条，使用现有正式评测入口，两路独立episode进程。结果4任务均完成，无ERROR：

| Task | Success / episodes |
|---|---:|
| move_stapler_pad | 2 / 2 |
| place_fan | 2 / 2 |
| place_mouse_pad | 1 / 2 |
| stamp_seal | 1 / 2 |

这只能证明这些初始seed可运行，不是50/200条完整验收。历史末次卡住的候选seed分别为
100006、100038、100008、100039，需按原顺序另行复测；不能以最初两条替代。

## 可选采集模式

复用`scripts/diagnostics/benchmark_robotwin_collection_pool.py`及现有持久化worker：

- `full`：原逐帧保存行为。
- `full_failures`：每条都逐帧读图、编码，最终只发布失败HDF5。
- `scout_replay`：scout仅在每48步重规划时取三路RGB，无逐帧数据输出；失败才完整重放。
- `paired_benchmark`：为测量基线成本，测试样本的成功/失败都重放一次；成功数据仅缓冲、
  不发布HDF5。不是正式10000条采集模式。

低频取图/渲染同步开关复用已部署FACT的
`evaluation/robotwin/model2robotwin_interface.py::_configure_robotwin_render`与
`_patch_robotwin_low_frequency_rgb`，不将这一适配冒称为RoboTwin官方现成失败采集器。
环境管理继续使用固定commit的RLinf/RoboTwin `robotwin/envs/vector_env.py`。
单环境单进程防止全局RNG被并行线程修改。

scout记录环境seed、原指令、结果、步数及耗时。diffusion seed沿用现有客户端公式：
`seed * 1000003 + control_step // 48`。测试暂存动作于内存，逐元素精确比较重放动作；
仅在动作序列、长度与终止结果一致时发布失败HDF5。不一致记录在JSON中，禁止冒充
原失败轨迹。每个失败HDF5包含逐控制步的三路RGB、state、action及真实末帧。

## 速度口径

对同一批候选，实际计算：

`T_scout_replay = sum(T_scout_all) + sum(T_replay_failed)`

`speedup = sum(T_full_all) / T_scout_replay`

成功率本身不足以确定收益：失败通常更长，需要分别测耗时。正式方案不会省去
FLUX每次重规划的RGB输入或20步动作去噪。种子预检、模型启动和并行调度耗时另计。
少量样本无法证明全部50任务都有相同速度或严格可重放。

测量目录：`outputs/scout_replay_benchmark_20260910`；尚未启动50任务×200条完整采集。

## 实测结果

`hanging_mug_pair/summary.json`：同120k actor，固定环境/指令/diffusion seed，
单推理服务、单持久环境，request batch1。配对测试两条，动作序列逐元素精确一致，
action_max_abs=0，replay_mismatches=0。仅失败样本发布HDF5，含901个观测和900个有效动作。

| Seed | 结果 | 控制步 | scout | 完整记录重放 | scout取图步数 |
|---|---|---:|---:|---:|---:|
| 200000 | 失败 | 900 | 76.162s | 220.086s | 19 |
| 200001 | 成功 | 315 | 18.140s | 102.475s | 7 |

scout+失败重放合计314.388s，对应逐帧记录基线322.561s，**1.026倍**。
配对基准本身为取得基线而多执行一次成功重放，实测wall442.574s（含启动），
不能把442.574s当成正式scout+失败重放用时。

按本次“一类成功/一类失败”的耗时结构，设成功率p：

`full = 102.475*p + 220.086*(1-p)`

`scout_replay = 18.140*p + (76.162+220.086)*(1-p)`

临界成功率约47.45%；低于它会变慢。成功率40%时约0.935倍；50%时1.026倍；
按历史46任务2109/2300=91.696%外推约2.72倍。该外推假设任务耗时结构相同，
并不能证明50任务真实吞吐；仅测2条，首条还包括冷启动/缓存影响。
各任务需使用独立的成功/失败时长统计再加权，不能把200条都按同一平均步数计算。

16项采集/数据writer测试通过，7项隔离seed/episode测试通过。没有修改FLUX、Q/V、
采样步数、reward、成功条件或控制动作，只新增显式基础设施诊断模式。
