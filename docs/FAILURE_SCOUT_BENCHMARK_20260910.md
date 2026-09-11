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
100006、100038、100008、100039，已按原顺序另行复测；不能以最初两条替代。

复测目录：`outputs/eval4_historical_seeds_20260910`。每任务从指定候选seed开始，
仅一次GPU尝试、不做CPU fallback。官方expert仍可能拒绝候选并递增seed。

| Task | 起始候选seed | 复测结果 |
|---|---:|---|
| move_stapler_pad | 100006 | 正常完成，policy失败（0/1），不再是ERROR |
| place_fan | 100038 | 正常完成，policy成功（1/1） |
| place_mouse_pad | 100008 | ERROR，exit134 |
| stamp_seal | 100039 | ERROR，exit134 |

后两者都在`get_obs → cameras.update_picture → left_camera.take_picture`抛出
`RuntimeError: vk::Queue::waitIdle: ErrorDeviceLost`，随后C++终止。该位置是环境取图；
不能把进程崩溃作为policy失败轨迹计入训练。50任务全部稳定跑满200条尚有这两个
已复现的渲染阻断项。本次没有更换渲染器、资产或驱动掩盖问题。

### 2026-09-11：seed 与渲染路径二次隔离

结论：**不是所有 seed 都崩溃，也不是故障起点的初始场景必崩；目前尚未修复完整 rollout 的 Vulkan 错误。**
注意候选起点不是实际评测 seed：`stamp_seal` 从100039开始，expert拒绝100039、100040、100041，
接受100042。不能把该次策略执行报错标为“实际场景100039崩溃”。

复用官方 `eval_policy.py::main` 构造任务，不加载policy、不改变相机/资产/渲染配置，
每seed使用独立进程取三次三路RGB：

| Task | 直接初始化测试的 seeds | 结果 |
|---|---|---|
| place_mouse_pad | 100000, 100001, 100006, 100007, 100008, 100009 | 6/6完成 |
| stamp_seal | 100000, 100001, 100038, 100039, 100040, 100041 | 6/6完成 |

再按官方 expert-check → close → setup 顺序测试故障起点，place_mouse_pad实际100008、
stamp_seal实际100042也都完成三次三路取图。上述只验证初始化/静态观测，**不是12条完整episode成功率**。
日志：190 `outputs/render_seed_probe_20260911/`。

完整120k action-only对照使用同一导出checkpoint、Clean环境、原48步规划和采样设置，未更改模型算法：

| 对照 | place_mouse_pad，起点100008 | stamp_seal，起点100039 |
|---|---|---|
| 原配置，GPU OIDN | exit134；完成5次三路观测后左相机报错 | exit134；完成2次三路观测后左相机报错 |
| 仅关闭 skip_action_render_sync | 仍exit134 | 仍exit134 |
| GPU失败后使用现有CPU OIDN fallback | GPU和CPU均exit134 | GPU exit134；CPU完成，实际100042，policy成功 |

对应目录依次为 `outputs/render_seed_full_eval_20260911/`、
`outputs/render_sync_control_20260911/`、`outputs/render_oidn_control_20260911/`。
每次均单环境/进程，前两组GPU4/5推理、6/7仿真；不占用GPU0–3的Stage2训练。
首次静态探针使用6/7，expert重置探针使用当时空闲的4/5。

CPU/GPU去噪可能改变RGB，进而改变闭环动作；一次CPU成功不能证明故障仅在CUDA去噪，
更不能证明两者轨迹等价。因此未默认启用CPU fallback、未跳过这些seed，亦未更改渲染器、
资产、控制动作或成功条件。错误episode不得作为正常失败数据进入回放。

只增强已有可选诊断：`ROBONANA_SAPIEN_TRACE_CAMERAS=1`记录相机姿态；
`ROBONANA_EVAL_DEBUG=1`记录异常发生时实际seed/控制步，并仅逐行跟踪官方评测文件，
不逐行跟踪模型/规划器。默认关闭，不改生产计算。相关本地测试14项通过。

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

新增异常退出保护后，采集/数据writer/隔离seed合计24项测试通过。没有修改FLUX、Q/V、
采样步数、reward、成功条件或控制动作，只新增显式基础设施诊断模式。
