# Stage 2 四卡续训与两个独立实验 / Four-GPU continuation and isolated probes

## 边界 / Boundaries

- GPU 0–3：Stage 2 从 step 4000 完整恢复，32/card × 4 × accumulation 2 = 256。
  10k endpoint、LR、数据、采样及 Value EMA 不重置；没有混入新的 10% 多任务数据。
- GPU 4–5：仅 Stage 1 action-only，固定100训练场景 + 新20个独立场景。
  这两类结果分开报告；复用 seed/指令，BF16、20步、shift1、48帧执行。
  按用户最新指令取消两个120k重复评测，保留此前采集的baseline记录。
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

入口 `scripts/diagnostics/train_action_student.py`，当前正式实验显式设置20000步、
batch104/card、accumulation1、global208、LR1e-4、warmup100、cosine20000；
每100步固定验证、每1000步保存。未将batch112设成适用于所有GPU的默认值。
从旧pilot的完整step1500保存点恢复学生和Adam，终点为总step20000（余18500步），
延长cosine预算并重算当前位置LR，不重置Adam、不沿用2000步末尾接近零的LR。
FP32 student/Adam masters，BF16 autocast；教师BF16且冻结；MSE为FP32。
通过 Accelerate DDP 联合训练，记录 W&B；没有教师EMA，没有 GT action loss。

`OneStepActionExpert` 位于独立模块 `models/flux2_action_student.py`。
48个噪声token读取冻结L/S/I逐层K/V，一次输出48×14动作；用相同噪声的20步教师
输出做stop-gradient监督。复用现有 ImageWAM-derived slim blocks、attention和
复制/缩放初始化，encoder/head新初始化。Q/V参数名与FLUX模型注册保持不变。

- MAC objective: https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L70-L101
- ImageWAM structure: https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a

按整个episode的稳定hash留出约10%，避免滑窗泄漏；沿用原四池采样。
实际150条中134条训练、16条留出：原始成功48/2、回放成功35/6、失败51/8。
整条留出意味着它的所有滑动窗口都不能进入学生训练，不是每个batch随机丢10%。
当前轻量验证每池最多取3条留出episode的中间窗口、总共最多8个状态，每状态8个
固定噪声；没有覆盖全部留出轨迹或其全部阶段。该小验证集不足以单独宣称方法有效。
验证集是“学生未训练过的状态”，不是“Stage1教师未见过的状态”，不能混淆。
固定独立验证噪声，记录MSE、teacher/student候选多样性RMS和比值；不只看train loss。
显式 `--action-student <step/model.safetensors>` 才启用隔离的 student policy；
必须匹配teacher SHA256，只支持action-only，不自动替换Stage2候选生成器。

所有新输出目录拒绝覆盖已有实验。失败pilot、训练产物与真实环境结果分开记录。

## 启动记录 / Launch record

- 36项相关回归测试通过，涵盖Q/V缓存数值、独立学生梯度、四卡配置、报告与脚本入口。
- 两卡真实smoke：`outputs/hanging_mug_fixed100_20260909/student_smoke`；完成2步、
  固定验证及student/Adam/RNG保存。不是正式学生效果评测。
- 正式学生：`experiments/hanging_mug_action_student_pilot_r2_20260910`；
  日志 `outputs/hanging_mug_fixed100_20260909/student_pilot_r2.launch.log`。
  首次pilot未显式传W&B entity，写入服务器凭据的默认团队，已停止；随后按用户
  明确要求删除错误run p9kz03sz、本地同名实验目录、启动日志和对应wandb目录。
  删除的是错误pilot，不是r2学生、教师、Stage1评测或Stage2。
  r2从新初始化独立学生开始；日志必须沿用源Stage1配置的
  `hongjia-liu-aalto-university`，缺少entity时拒绝启动，不能静默使用默认团队。
  原2000步pilot已停止，最新完整checkpoint为1500，约1501–1910的未保存更新未沿用。
  当前20k训练未开启`--launch`，不自动用早期学生跑真实环境；先检查拟合和多样性。
- Stage1配对：`outputs/stage1_paired_eval_r2_20260910`；
  初次预检因缺少显式SAPIEN渲染设备而退出，r2已补齐，原清单与checkpoint均未修改。
  旧调度父进程已取消，新调度使用`--stage1-only --adopt-running-fixed100 1006623`，
  不重启当前Stage1采集，接着跑独立20场景。两个120k任务均未开始。
  报告分别在`fixed100_stage1/selected_world/index.html`、
  `heldout20_stage1/selected_world/index.html`，每组完成后生成；图片/JSON边跑边保存。
- 当前20k学生：`experiments/hanging_mug_action_student_bs104_20k_20260910`；
  日志`outputs/hanging_mug_fixed100_20260909/student_bs104_20k.launch.log`。
  支持同两卡学生/Adam恢复，旧pilot scheduler按原公式重建，新checkpoint保存scheduler。
  变更batch后不是逐位相同的数据和随机噪声轨迹。
- 两卡完整前向/反向/Adam三步实测：batch48峰值allocated83.52GiB/reserved84.04GiB；
  batch112峰值allocated169.70GiB/reserved172.20GiB。不是只测教师推理。
  但正式恢复Adam、执行验证后，112首个训练反向OOM（进程178.28GiB，碎片/空闲
  reserved7.50GiB，再申请558MiB失败），未产生新更新。因此短测112不是稳定上限，
  正式改为104/card以留出验证切换和allocator余量，不改精度或模型。
- Universal只用于加载转换目录；后续从新保存的原生四卡checkpoint恢复时，要使用
  原生DeepSpeed配置，不要对原生checkpoint继续开启Universal标志。
