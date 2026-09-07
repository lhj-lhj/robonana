# MAC 三个训练阻断项修复与验证（2026-09-06）

> 本文保留当时的验证精度和结果，不作为当前启动配置。
> 维护版本 FLUX/Q/V 仅支持 FP32，冻结外部编码器不改。

本文是 [首次就绪审查](MAC_READINESS_REVIEW_20260906.md) 的后续记录，
只覆盖失败尾段监督、ZeRO 保存/恢复协议、多卡非有限 loss 保护。
不是 world model/critic 收敛或完整环境闭环的验收报告。

## 1. 固定 48 步滑窗

设一条轨迹有 T 个真实 action 和 T+1 个 observation：

- 成功：起点为 `0..T-1`，共 T 个窗口，最多补 47 步吸收态。
  future state/image 使用成功后的末帧；成功后的 reward 为 0，
  `success_terminal_h=1`。补齐 action 重复最后一个真实 action 作为
  conditioning 占位，但 `action_valid_mask=0`，不计算这些位置的 BC。
- 失败：起点仅为 `0..T-48`，共 `max(T-48+1,0)` 个完整窗口。
  不做 padding；最后一个窗口的 future observation 正好是轨迹末帧。
  timeout 不当作成功终止，不能因为超时就将 bootstrap 清零。
- 不再从没有 outgoing action 的最终 observation 创建 MAC 训练样本。
- 中间缺少真实 transition 时明确报错，不把不完整窗口冒充 48 步监督。
- LeRobot 和 HDF5 共用合法窗口索引，episode/pool sampler 使用窗口数。
  `legacy_v1` 的 h_idx 起点和 loss 规则不变。

实现：`src/robonana/data/robotwin_hdf5.py`、
`src/robonana/training/losses.py::masked_action_mse`。
第一阶段继续成功样本训 BC、全部样本训 world loss。
190 实际数据入口的 `data_size` 从旧的 observation 计数修正为 56,773。

## 2. DeepSpeed 保存与严格恢复一致

FACT 保存时传入 `exclude_frozen_parameters=True`，但恢复使用严格加载，
原先 critic 的 ZeRO module payload 因此缺少冻结 FLUX。

新增 `full_deepspeed_checkpoint` 薄适配器：仅在 FACT 同步保存调用期间，
将 DeepSpeed 的 `exclude_frozen_parameters` 强制为 False。保存完成或异常
时均恢复原方法。继续复用 FACT 的 optimizer/scheduler/RNG、导出和 hooks，
不修改 vendor、不使用 `strict=False`，非 DeepSpeed 后端不受影响。

两个阶段都保存完整 module，包含当阶段冻结的权重。代价是 checkpoint
变大，但不引入 EMA FLUX 或第二个驻留 FLUX 模型。

Value 生命周期保持既定原则：新一轮继承 online Value，新的 critic 阶段
用当前 online Value 精确初始化 target Value、计数置零；只有同一 critic
训练恢复才读取该 checkpoint 的 target Value 和计数。缺少 EMA 附件时
拒绝冒充完整恢复。Q 不创建 EMA。

**旧 checkpoint 不自动修复。** 此前已省略冻结参数的 ZeRO checkpoint
仍不满足严格恢复协议。其完整 transformer 导出可用于权重热启动，
但不能据此宣称 optimizer/scheduler/RNG 已被精确续接。

## 3. 全局 nonfinite 检查与同步停训

每个 micro-step、进入 backward 前计算：

```text
local_bad = int(not isfinite(local_loss))
global_bad = SUM(local_bad over all ranks)
if global_bad > 0: raise FloatingPointError on every rank
```

只使用 Accelerate 支持的 SUM，不再使用会退化成求和的 `min`。
任意 rank 出现 NaN 或 Inf 时，整组在该 micro-step 的 backward、optimizer、
scheduler、Value EMA 前退出。不跳过一个 backward 后继续使用未完成的
DDP reducer/ZeRO 累积状态；应从最后一个完整 checkpoint 重启。

这是非有限 **loss** 的保护，不是“有限 loss 永远不会产生异常 gradient”
的保证，也不替代长期梯度、Q/V 数值及 world-model 质量监控。

## 190 验证

环境：`/data3/hongjia/robonana`，Python 为
`/data3/hongjia/conda/envs/robonana/bin/python`。

- 完整 pytest：`182 passed, 1 skipped`（70.44 秒）。
- 真实双进程 CPU/Gloo：NaN/Inf 分别注入第 1、2 个累积 micro-step，
  四项通过，每个 rank 都中止；参数、scheduler、EMA 均未继续更新。
- 真实双卡 DeepSpeed ZeRO-2/BF16：同样四项异常注入全部通过。
- TinyCritics 双卡 ZeRO 严格恢复：主动破坏 online 参数（含冻结模块）和
  target Value 后，恢复 module、optimizer、scheduler、RNG、EMA 及计数；
  恢复后下一步与未中断参考路径的 module/EMA **逐元素完全一致**。
  这里使用非零学习率，覆盖 Adam moments 和 ZeRO FP32 master weights。
- 真实 Klein-4B 第一阶段：120k 迁移、真实数据训练 1 步并保存成功；
  重启后严格恢复所有训练状态成功，exit=0。第一阶段此次恢复为加载检查，
  没有继续执行额外训练步。
- 真实 Klein-4B 第二阶段：从第一阶段权重训练 1 步、保存；独立重启后
  严格恢复成功，日志确认 target Value `updates=1`，继续执行第 2 步后
  `update_count=2`，再次保存并正常退出。两个阶段都使用实际数据、双卡
  ZeRO-2、每卡 batch=2、梯度累积=2。

真实 4B 短测的 `max_steps=1,warmup=0` 导致保存时 scheduler 已把 LR 降为 0，
因此 critic 恢复后的第二步用于检查执行流程和 EMA 接续，不宣称验证了
非零学习率的参数更新等价性；后者由上述 TinyCritics 非零 LR 对照验证。
这些 1–2 步的 world/critic 权重不能用于判断策略收益或正式评测质量。

复现分布式隔离检查：

```bash
cd /data3/hongjia/robonana
export PYTHONPATH=src:third_party/FACT:third_party/flux2/src
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1
PYTHON=/data3/hongjia/conda/envs/robonana/bin/python

for bad in nan inf; do
  for micro in 0 1; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/validate_mac_distributed_safety.py \
      --backend deepspeed --mode nonfinite --bad "$bad" --micro "$micro"
  done
done

# 必须选择一个不存在的新目录，脚本拒绝覆盖已有产物。
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/validate_mac_distributed_safety.py --backend deepspeed --mode resume \
  --checkpoint-dir /data3/hongjia/robonana/_tmp/safety_zero_resume_new_run
```

实际诊断产物与日志（不提交 Git）：

```text
_tmp/safety_zero_resume_20260906_b/
_tmp/safety_world_20260906_a/logs/train_20260906T064813Z.log
_tmp/safety_world_20260906_a/logs/train_20260906T065428Z.log
_tmp/safety_critic_20260906_b/logs/train_20260906T065411Z.log
_tmp/safety_critic_20260906_b/logs/train_20260906T065704Z.log
```

早先 `safety_zero_resume_20260906_a` 的诊断脚本新目录检查有多 rank 竞态，
已改为全局归约后才保存；`safety_critic_20260906_a` 启动命令误传了权重
目录而非 `.bin` 文件，未进入训练。两者不是修复后恢复协议失败，残留
诊断目录保留，没有删除用户已有产物。

本轮诊断目录约 65 GiB（world）和 41 GiB（critic），保留供检查。
critic 测试运行自带的 `checkpoint_total_limit=1` 自动移除了本轮新建的
step-1 checkpoint、保留 step-2；step-1 文件未另行备份，训练日志仍保留。
没有删除此前已有的实验或诊断 checkpoint。验证结束时 8 张 GPU 均无显存占用。

未启动正式长实验、未写入正式 replay。首次审查的其他 P2 运维问题、
M=32 环境成功率以及 selected-policy -> 新数据 -> 下一轮 BC 的闭环验收
不在本次修复验证范围。
