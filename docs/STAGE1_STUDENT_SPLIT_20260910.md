# Stage 2 四卡续训与两个独立实验 / Four-GPU continuation and isolated probes

## 边界 / Boundaries

- GPU 0–3：Stage 2 从 step 4000 完整恢复，32/card × 4 × accumulation 2 = 256。
  10k endpoint、LR、数据、采样及 Value EMA 不重置；没有混入新的 10% 多任务数据。
- GPU 4–5：Stage 1 与 120k action-only 对照，固定100训练场景 + 新20个独立场景。
  这两类结果分开报告；复用 seed/指令，BF16、20步、shift1、48帧执行。
- GPU 6–7：冻结 Stage 1 教师，两卡联合训练独立一步学生。暂不接入 Q/V。

## Stage 2 恢复 / Restore

源：`experiments/hanging_mug_fixed100_round0_fresh3000_s1_10k_s2_10k_20260909/critic/models/checkpoint_epoch_13_step_4000`。
使用 `scripts/diagnostics/prepare_universal_checkpoint.py` 调用已安装 DeepSpeed 官方
`deepspeed.checkpoint.ds_to_universal`，输出至新的 checkpoint_views，原文件不变。
普通 BF16 loader 不会自动8→4重分片。Universal 加载后 fused Adam 的 step 标量
需要放回参数设备；现有 Trainer 的 resume 中做窄范围校验，不重建 moments。

`scripts/run_robotwin_train.sh --config robonana.configs.critic_continuation.config`：
`ROBONANA_GPU_IDS=0,1,2,3`、`ROBONANA_BATCH_SIZE_PER_GPU=32`、
`ROBONANA_GRADIENT_ACCUMULATION_STEPS=2`、`ROBONANA_MAX_STEPS=10000`、
`ROBONANA_UNIVERSAL_CHECKPOINT=1`。设置独立 project/resume config/checkpoint 路径。
首次试跑失败于 CPU Adam step；第二次恢复审计确认 Adam=4000、EMA=4000、
LR=7.008477123264848e-5，已实际更新至4010，约23.7秒/optimizer step。
运行目录：`experiments/hanging_mug_critic4000_4gpu_acc2_r2_20260910`。
改变卡数/累积不会保证后续随机样本与八卡逐位相同。

## Stage 1 配对评测 / Paired evaluation

入口 `scripts/diagnostics/compare_stage1_policy.py`，调度现有 RLinf pool collector，
并复用 `report_selected_world_eval.py`。不复制仿真/推理实现，不自动加入训练回放。
Stage 1 权重是上述 fresh run 的 world_policy step10000；baseline 使用保留的
`checkpoints/120k_action_only_export_20260909`。120k 的 world heads 未训练，不绘制
其预测并冒充有效结果；Stage1 保存每个执行chunk对应的预测与真实未来、reward/success。
Action-only 的 Q 为 null，而不是伪造为0或重新计算无关 Q。
100固定seed取 `outputs/hanging_mug_fixed100_20260909/seeds.json`；独立场景从300000
起由官方 expert feasibility 检查选20个，不按policy成功筛选。

## 一步学生 / One-step student

入口 `scripts/diagnostics/train_action_student.py`，默认2000步pilot、batch8/card、
accumulation1、global16、LR1e-4、warmup100、cosine2000；每100步验证、500步保存。
FP32 student/Adam masters，BF16 autocast；教师BF16且冻结；MSE为FP32。
通过 Accelerate DDP 联合训练，记录 W&B；没有教师EMA，没有 GT action loss。

`OneStepActionExpert` 位于独立模块 `models/flux2_action_student.py`。
48个噪声token读取冻结L/S/I逐层K/V，一次输出48×14动作；用相同噪声的20步教师
输出做stop-gradient监督。复用现有 ImageWAM-derived slim blocks、attention和
复制/缩放初始化，encoder/head新初始化。Q/V参数名与FLUX模型注册保持不变。

- MAC objective: https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L70-L101
- ImageWAM structure: https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a

按整个episode的稳定hash留出约10%，避免滑窗泄漏；沿用原四池采样。
验证集是“学生未训练过的状态”，不是“Stage1教师未见过的状态”，不能混淆。
固定独立验证噪声，记录MSE、teacher/student候选多样性RMS和比值；不只看train loss。
显式 `--action-student <step/model.safetensors>` 才启用隔离的 student policy；
必须匹配teacher SHA256，只支持action-only，不自动替换Stage2候选生成器。

所有新输出目录拒绝覆盖已有实验。失败pilot、训练产物与真实环境结果分开记录。
