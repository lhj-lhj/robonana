# Model boundary and upstream references

RoboNana reuses the official FACT/FLUX.2 implementation and adds only the
fixed-48 RoboTwin wiring. The maintained model is `MacFlux2FACTModel`.

| Source | Reused | RoboNana extension |
|---|---|---|
| FACT | trainer lifecycle, samplers, collators, Accelerate/DeepSpeed | fixed-48 RoboTwin data and losses |
| FLUX.2 | projections, RoPE, modulation, transformer blocks | action/state/world token layout |
| MAC | one imagined rollout, Q action selection, Value-only target | RoboTwin reward/world wiring |
| ImageWAM | cached MoT K/V expert structure and initialization policy | deterministic scalar Value/Q experts |

References: [MAC](https://github.com/kwanyoungpark/MAC), [ImageWAM](https://github.com/yuyangalin/ImageWAM/tree/5d4a341ed20a95cdb08f0293f3d44778b9a9e05a).

## Fixed-48 information flow

```text
[L | S | I | A_pred | G_clean | R[48] | success | S' | I']
```

The world cascade is `R -> success -> S' -> I'`. The action track is not
visible to world targets except the clean action conditioning track `G`.
Value reads only `[L,S,I]`; Q reads `[L,S,I,G]`. Both are deterministic,
one-query experts. FLUX is frozen during critic training. Q has no target
network; only Value is Polyak-averaged, stored/updated in FP32 with BF16
autocast target forward on the shared BF16 FLUX cache.

## Checkpoint boundary

Runtime loading accepts complete `mac_mot_v2` checkpoints. The current 50-task
experiment starts from its own pretraining run; see `MULTITASK_MBRL_PROTOCOL.md`
for exact paths. The older single-task config's default is not this experiment's
initialization. A new critic phase copies the online Value expert into its EMA;
resuming the same critic phase restores the saved EMA.

`scripts/data/convert_120k_action_checkpoint.py` is an explicit one-time export
for the archived actor. The existing export is action-only; its newly initialized
world/critic heads are not trained. The legacy forward, `SegmentMap` and
`build_attention_bias` have been removed; MAC reuses the base modules/helpers.
Conversion is checked against CPU outputs recorded before that removal.

## V3：独立 Action expert（2026-09-24）

`MacFlux2FACTModel(architecture_version="mac_mot_v3")` 复用相同训练、采样、评测入口。
共享 FLUX 序列为 `[L,S,I,G,R,U,S',I']`；`MacSegmentMap.pred_action` 是空切片，
输出的 `action` 仍为 `[B,48,action_dim]`，由独立 `FlowActionExpert` 产生。
V2 的 `[L,S,I,A,G,R,U,S',I']` 和权重名不变。

- Action expert 复用 ImageWAM 派生的 slim double/single blocks、AdaLN head 和
  FLUX 插值初始化；输入 noisy action 与真实 flow timestep，输出 velocity。
- 每层先取得共享 FLUX 的 C（语言、当前 state、当前图像）K/V，再计算 Action
  对 `[C,A]` 的单次 softmax。联合训练不 detach、不重跑一次 backbone。
  Action loss 可回传共享 FLUX；Action 不读 G 或任何未来标签，World 不读 A。
- 推理用已有 C cache 和 20 步 Euler；候选映射按层展开 K/V。
- `action_in` 留给 clean action G；v3 不保留共享 `action_out`。
- Action expert 全部参数归入现有 `robot_modules`，与 Q/V 使用同一 robot LR，
  不新增第三组 LR。Stage2 冻结 Action/World，仅训练 Q/V。
- 新配置必须显式声明 architecture_version 和 expert_hidden_dim；v3 首轮只支持
  fixed48。严格加载记录的架构，不自动把 v2 checkpoint 转为 v3。

`configs/train_v3.json` 是首轮模板：原始 FLUX、fixed48、现有吸收态数据逻辑、
120000 步、global256（8×16×2）、GC 开启、峰值 LR 2e-5/1e-4、20 步动作采样。
路径需按服务器资产配置；这是模板，不代表已启动或已测量生产显存/吞吐。
用相同配置仅将 architecture_version 改为 mac_mot_v2，即为控制变量对照。
