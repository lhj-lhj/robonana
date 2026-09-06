# Hanging-mug 初步实验：world 拟合 -> MAC 收益检查

用户确认第一阶段预算至少 5,000 步。本轮任务是 `hanging_mug`（历史对话中
也写作 hugging mug），服务器为 190，源码目录 `/data3/hongjia/robonana`。

## 首先启动的实验

- 独立目录：`experiments/hanging_mug_mac_pilot_20260906/world_policy`。
- GPU 6、7；每卡 batch=4、累积=2，global batch=16；seed=20260906。
- 从当前 1,000-step `mac_mot_v2` checkpoint 启动；不再在运行时迁移 120k 权重。
- world/policy 5,000 步，warmup=250；backbone LR=2e-5，robot LR=1e-4。
- 使用原始成功 50 条及当前 replay（成功 5 条、失败 45 条）；当前空历史
  失败池的权重转给最新失败池，成功/失败采样合计约 50%/50%。
- 成功 BC+world，失败仅 world；固定 48 步、失败不 padding。
- 保存 500/1000/2000/3000/4000/5000，limit=8，保留本轮全部检查点。
- `WANDB_MODE=offline`，持久文本日志及本地 W&B 记录；不要求外部服务在线。

配置为 `robonana.configs.robotwin_flux2_4b_mac_pilot.config`，只覆盖实验预算、
warmup、保留数量，继续使用当前维护的模型、数据和 FACT trainer。

`scripts/start_mac_world_pilot.py` 拒绝覆盖已有实验目录，依次运行：

1. 保存 `pilot_config.json`、源码 SHA256 清单和 source commit。
2. 1,000-step MAC 权重的固定窗口 `probe_initial` 基线。
3. 真实两卡 world/policy 训练 5,000 步。
4. `probe_step5000`；完成后停在 `world_complete_review_required`，不因训练
   结束就无条件启动 critic。

状态：`pilot_status.json`；训练：`logs/train_*.log`；固定窗口指标：
`probe_initial/metrics.json` 与 `probe_step5000/metrics.json`。
中途检查可以对已完成保存的 checkpoint 单独运行 `probe_mac_world_fit.py`，
建议用空闲 GPU 4，不干扰训练 GPU。确认保存完成及下一训练步出现后再读取。

## 如何判断 world 是否拟合

`probe_mac_world_fit.py` 复用 critic imagination 的 `sample_mac_world`：
输入真实 clean action，从纯噪声执行 20 步 Euler；不给 future GT 做输入。
每个非空池固定取 4 条轨迹的首/中/末合法窗口，共 12 个窗口。原始成功、
收集成功、最新失败各单独报告。窗口及噪声在 checkpoint 间一致。

检查 future latent MSE、归一化 future-state MSE、48-reward BCE/accuracy、
success BCE/probability，并与“直接复制当前图像/状态”的 persistence baseline
比较。保存预测 latent/state，可进一步解码为预测/真实图像对照。
失败数据 reward/success 几乎为常量，所以不能仅靠分类准确率判断 world 可用。
还要检查真实成功窗口，防止模型全预测失败。

这是训练集拟合实验，**不是 held-out 泛化**。GT action 上拟合改善也不证明
world 对 policy 新采样 action 的预测可靠；RL 阶段必须独立验证环境结果。

## 后续 critic 与环境对照

先检查 world 结果。若有有限、实际的生成误差改善且无明显退化，再从
选中的完整 world checkpoint 新启动独立 `critic` 目录：

- 同一 pilot config，`ROBONANA_MAC_PHASE=critic`、initialization=trained。
- 显式指定该 world `.bin` 文件及对应 `config.json`；不恢复 world optimizer。
- 冻结 FLUX，Q/V 训练 500 步，LR=1e-5、warmup=100、H=1、训练 M=8。
- 新阶段 target V 精确复制继承的 online V；仅 V 做 EMA，不引入 target Q。
- 固定 backbone/policy 的同一 critic checkpoint，对 M=1 与 M=32 各评测
  20 个 episode；相同 `ROBONANA_EVAL_SEED_GROUP=21`，任务 hanging_mug、demo_clean。
- 用 `eval_robotwin_all_tasks_parallel.sh` 做两组只评测的独立输出。
  先不运行全轮 collect/prepare 脚本，避免初步对照向既有 replay 追加数据。

观察 Q/V 尺度、候选 Q 分布和实际成功数，不能把 critic loss 降低当成策略收益。
20 episodes 仅筛查初步信号，不作为显著提升结论。若 world 或 critic 发散，
保存证据并停止该阶段，不自动增加训练预算、改公式或改 replay。
