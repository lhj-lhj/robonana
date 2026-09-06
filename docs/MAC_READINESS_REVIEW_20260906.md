# mac_mot_v2 正式实验就绪审查（2026-09-06）

## 结论

**暂不通过正式长实验验收。** 两阶段 fresh-start 训练能启动和保存，
但真实 DeepSpeed critic 断点恢复失败；非有限 loss 的跨卡保护有错误；
失败 timeout 尾段存在固定 48 步 world-model 监督错位。

本次仅修改用户明确要求的 BF16 GEMM 低精度中间累加开关，增加回归测试。
下列其他问题是审查发现，**尚未修复**。没有启动正式长训练，没有向正式
replay 添加数据，也没有用短测 checkpoint 替换正式预训练权重。

审查覆盖当前维护路径：120k 迁移、MoT 初始化及信息流、前缀复用、两阶段
训练/损失/EMA、FACT/Accelerate/DeepSpeed 集成、数据池、环境推理入口、
rollout writer、收集/准备/跨轮脚本和相关测试。旧 h_idx 路径只审查兼容边界，
没有重新做旧版完整训练验收。本次没有进行新的 RoboTwin 环境成功率评测，
因此也没有完成当前版本的 selected-policy -> 新轨迹 -> 下一轮 BC 实机闭环验收。

## 实测证据（190）

工作目录：`/data3/hongjia/robonana`；模型环境：
`/data3/hongjia/conda/envs/robonana/bin/python`。

- 全量测试：`163 passed, 1 skipped`，17.27 秒。
- 生产模型入口已设置
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False`。
  BF16 存储和 autocast 保留。B200/真实 120k、B=1、10 Euler steps、M=1/8/32
  的 synthetic full/cached 对照中，action 和 Q 最大绝对差均为 0。
  M=32：full 4.120 秒，cached 0.795 秒，cached 峰值 11.04 GiB。
  这些是指定输入的结果，不是全输入逐位相同或成功率提高的保证。
- GPU 6/7、每卡 batch=2、累积=2，第一阶段真实数据训练两步成功；
  加载 153 个兼容 tensor，跳过 9 个旧 tensor；峰值 allocated 48.557 GiB。
- 从上述 checkpoint 接续第二阶段两步成功；只训练 1,126,711,808 个 Q/V
  参数，EMA Value 563,355,904 个参数，更新次数 2；峰值 22.193 GiB。
- 第二阶段恢复并继续第 3 步：**恢复时报 missing FLUX keys，exit=1**。
- 真实数据池：original success 50 条；collected success 5 条；historical
  failure 0 条；latest failure 45 条。有效采样权重为 25%/25%/0%/50%。
- 双进程 Accelerate 归约诊断：两个 rank 输入 finite flag `[1,0]`，
  `reduce(reduction="min")` 两边都输出 `1`，错误地接受该步。

两步短测使用 warmup=0、max_steps=2，只用于接口和状态验证。第二阶段
第 2 步 online-next-V 均值约 -3609、candidate-Q 均值约 -4310；这说明
短测权重不可当成可用 critic，**不代表已经证明正式 warmup 配置必然发散**。
正常奖励为 [-1,0]、gamma=.999 时，真实无限折扣回报范围是 [-1000,0]；
无约束回归输出不必自动处在这个范围，应监控越界和 Q 排名质量。

日志/产物（全部是本次独立诊断目录）：

```text
_tmp/readiness_20260906_b/logs/train_20260906T055557Z.log
_tmp/readiness_20260906_b_critic/logs/train_20260906T055859Z.log
_tmp/readiness_20260906_b_critic/logs/train_20260906T060236Z.log
_tmp/readiness_probe_20260906.py
```

第一阶段产物约 60 GiB，critic 约 27 GiB；保留供复现，没有删除用户文件。
验证进程已退出，结束检查时 8 张 GPU 都无显存占用。

## P1：正式实验前必须处理

### 1. ZeRO checkpoint 的保存/严格恢复协议不一致

位置：`src/robonana/training/robotwin_trainer.py:572`、`:524` 调用的 FACT
生命周期；190 上 `third_party/FACT/fact_train/trainers/trainer.py:790`、`:515`。

FACT 使用 `accelerator.save_state(..., exclude_frozen_parameters=True)`，
但 resume 使用 `load_module_strict=self.checkpoint_strict`，默认严格。
critic ZeRO payload 只有可训练专家，恢复却要求冻结 FLUX 的全部 key。
本次完整保存后重启就复现，missing keys 从 `img_in.weight`、`time_in.*`
一直到 actor/world heads。单独保存的完整 transformer 文件没有在该
DeepSpeed 恢复路径中自动补齐这些 key，EMA load hook 尚未执行就失败。

应统一保存/恢复协议：例如保存完整 ZeRO module state，或先严格重建和
验证冻结部分，再仅允许准确已知的 frozen keys 缺失。不要无条件用
`strict=False` 隐藏所有缺失。补真实 ZeRO 双卡的两个阶段保存/恢复测试，
检查在线参数、optimizer、scheduler、RNG、EMA 参数和计数。

### 2. 非有限 loss 的跨卡检查使用不支持的 min

位置：`src/robonana/training/robotwin_trainer.py:580`。

190 的 Accelerate reduce 只支持 mean/sum/none，底层执行 SUM。
任意一张卡坏掉而另一张正常时，当前代码仍认为可训练；随后 FACT 对
本 rank 的 NaN 单独 return，其他 rank 进入 backward，可能 collective
不匹配；Inf 也可能继续传播。应改用受支持的归约计算全局 bad flag，
同时确保所有 rank 和累积 micro-step 一致地中止/跳过，不能只改字符串
后就假定 DDP/ZeRO 的 reducer 状态安全。补 NaN/Inf 单卡注入测试。

### 3. timeout 尾段被当成完整 48 步 future 监督

位置：`src/robonana/data/robotwin_hdf5.py:567`、`:602`；
`src/robonana/training/robotwin_trainer.py:1006`、`:1010`、`:1018`。

`future_index=min(t+48,last)`，reward 的未知后缀被 mask，但 future
state/image 和 success loss 没有对应有效性处理。实测失败 episode0
长度 901，t=891 时 h=48、delta=9，future=900，reward 仅 9 个位置有效，
而这个 9 步后的 state/image 仍监督“48 步后”，success 仍监督 0。
timeout 不是 absorbing terminal，不能把截断后的未知未来当成静止状态，
也不能断言未来完整 48 步绝不会成功。这会把错误 world 监督传到 imaginary TD。

固定 horizon 不必改：失败尾段可仅训练已知 reward 前缀，mask 未知
future/success；或者仅为 world 的完整 future 目标抽取完整 48 步片段。
成功 terminal 的 absorbing 后缀则另行保留。最后无 outgoing transition
的 observation 也要有明确采样/监督规则。

## P2：闭环与复现风险

### 4. 最新失败池为空时，下一轮无法训练

位置：`src/robonana/configs/posttrain_config.py:45`；
`src/robonana/data/robotwin_hdf5.py:879`。

latest_failure 不允许为空，sampler 只处理 historical-failure 和
collected-success 为空。某轮全部成功是合法情况，却会直接报
`posttrain pool 'latest_failure' contains no episodes`，即使旧失败和成功
数据都够。应定义 latest 为空时的重分配策略，并覆盖两个失败池都为空。
当前 round0 有 45 条失败，所以本次启动没有触发。

### 5. checkpoint 更新不是原子替换，且只保留一个

位置：`src/robonana/configs/robotwin_flux2.py:205` 与 FACT save_checkpoint_step。

继承的 `checkpoint_total_limit=1` 配合 FACT 在保存前清理旧 checkpoint，
使进程在大文件写入期间中断时可能同时失去上一份完整状态和新状态。
两步 world 保存实际耗时约 87 秒，并非很短的窗口。应先写临时 checkpoint，
全 rank 保存成功/完整性确认后再发布并清理旧版本。

### 6. 跨轮完成标记没有绑定配置或 checkpoint 身份

位置：`scripts/run_hanging_mug_mac_round.sh:88`、`:198`。

`.done` 和精确 step 文件名决定是否跳过阶段/评测。重用 run_root/project_dir，
但更换 source checkpoint、采样设置或代码时，会复用旧训练和旧评测结果。
评测 episode ledger 也应绑定 checkpoint/config，而不只是 seed/episode 数。
应保存来源及配置摘要，恢复时校验不匹配就明确拒绝；只改目录名是操作规避，
不是程序级保证。

### 7. 自定义配置未完整贯穿训练和环境入口

位置：`scripts/run_hanging_mug_mac_round.sh:18`、`:92`、`:179`；
`scripts/eval_robotwin_all_tasks_parallel.sh:256`。

- `ROBONANA_INITIAL_DATASET_ROOT` 用于评测/收集，但没有传成训练的
  `ROBONANA_DATASET_ROOT`。只设置前者时训练仍读默认数据/stats，推理却
  使用自定义 stats，可能造成归一化不一致。
- imaginary 的 `ROBONANA_MAC_SAMPLING_STEPS/FLOW_SHIFT` 可修改，而评测
  固定 20 steps，flow_shift 用默认 1。当前默认值是一致的；修改后会静默分叉。

应由同一份经过验证的运行配置生成训练、评测、收集参数和 provenance。

### 8. 小全局 batch 会永久丢掉某些采样池

位置：`src/robonana/data/robotwin_hdf5.py:898`。

pool 数量按每个 batch 固定向下取整再用固定顺序补余数。实测本轮权重下：
global batch=2 时 counts=[1,0,0,1]，collected success 永远不进 BC；
global batch=1 时只有 latest failure。此处 FACT 传入的是每卡 batch ×
卡数 × 梯度累积数；默认 4×2×2=16 不受此问题影响。应按跨 batch 配额
累积/随机采样维持长期比例，或验证并拒绝不支持的 global batch。

### 9. 成功 replay 没有 collection round 上界

位置：`src/robonana/configs/posttrain_config.py:54`。

历史失败限制 `< current_round`，最新失败限制 `== current_round`，
collected success 却没有 `round_max`。向累计 replay 添加新轮成功轨迹后，
重新跑早期 round 会看到未来轮成功数据，污染可复现实验。当前顺序向前
运行且根目录里没有未来数据时不触发。应限制成功回放 `round_id <= current_round`。

## 已核对且未发现本轮新的结构性问题

- one FLUX、online Q/V、Value-only FP32 EMA；没有 Q target / EMA FLUX。
- 新 critic 阶段 target 从继承的 online Value 复制；同阶段 resume 的
  EMA hook 要求两份文件齐全。该设计合理，但被上述 ZeRO 集成错误挡住。
- query 和 scalar head 新初始化；专家 body 从固定 ImageWAM 规则复制/缩放。
- C 前缀不读候选动作，Q 读 C/G；Value 只读 C。cached Q selection 不再
  额外计算被丢弃的 Value，也不跨 candidate 泄漏。
- H=1；R 从 48 个二值 reward logits 解码并折扣求和；V target 用 EMA V，
  Q target 用 stop-gradient online V；return_scale 的乘除位置一致。
- failure action BC mask 存在；writer 保存实际执行 action、selected-Q 元数据
  和 reset-pre final observation。环境闭环仍需新版本真实运行验收。

## 修复后的正式验收顺序

1. 先修 P1；执行真实 ZeRO 保存/恢复，以及单 rank NaN/Inf 注入。
2. 处理 replay/运行配置/保存协议边界；至少完成两轮最小闭环。
3. 固定独立 episode 级 validation 集；验收 world 的 48 个 reward logits、
   success 校准、future-state 误差和 future-image 质量，而不是只看 train loss。
4. world 通过后做受控 critic pilot，监控 Q/V 范围、TD target、Q margin、
   排序与真实结果关系。当前默认训练步数不是 world 达标的保证。
5. 用同一训练后 checkpoint、配对 seed 做 M=1/M=32 真实环境对照，核验
   selected-policy 成功数据确实进入下一轮 BC，再启动正式多轮实验。

## 版本边界

本次审查基于本机 `e4dbbac` 加已有工作区修改，190 是旧 Git HEAD 加同步的
工作树，不应把 Git HEAD 单独当作被测版本。关键文件 model/测试/trainer/
posttrain config/round launcher 的 SHA256 已逐一核对本机与 190 一致。
本次 focused commit 不打包此前尚未提交的 README、target 生命周期等改动。
正式实验前应把完整运行版本收敛为可复现 commit/manifest，并记录依赖版本。
