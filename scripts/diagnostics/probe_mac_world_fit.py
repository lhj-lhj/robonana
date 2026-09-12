# 中文：诊断：用训练集固定窗口检查 world model 重建能力。
# English: Diagnostic: probe world-model reconstruction on fixed training windows.
# 调用 / Invocation: 写重建图和指标，不训练；不能当作泛化成功率。 / Writes reconstructions/metrics without training; not a generalization success-rate evaluation.
# 导航 / Guide: scripts/README.md (diagnostics)
"""Fixed real-action, pure-noise MAC world probes, separately by replay pool.

This checks TRAINING-SET fitting, not held-out generalization or policy success.
Uses the same sample_mac_world implementation as critic imagination; no custom
world rollout or teacher-forced future image/state. Saves per-window metrics
and predictions so that failures cannot be hidden by a mixed-pool mean.
"""

import argparse
import json
import html
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robonana.data.robotwin_hdf5 import RoboTwinHDF5Dataset
from robonana.data.robotwin_lerobot import RoboTwinLeRobotDataset
from robonana.models.pretrained import load_flux2_fact_trained_checkpoint
from robonana.sampling import sample_mac_world
from robonana.sampling import flow_euler_schedule
from robonana.training.continuation import restore_config_tuples


def configured_pools(config):
    """Support both the single pretraining dataset and MAC replay pool lists."""
    pools = config["dataloaders"]["train"]["data_or_config"]
    return pools if isinstance(pools, (list, tuple)) else [pools]


def probe_indices(dataset, episode_count=4):
    """Reproducible episode-spread early/middle/final starts, including success tails."""
    len(dataset)  # Build the same legal fixed-48 index as training.
    result = []
    for episode in np.linspace(0, len(dataset.records) - 1,
                               min(episode_count, len(dataset.records)), dtype=int):
        start, stop = int(dataset.episode_starts[episode]), int(dataset.episode_stops[episode])
        result.extend(sorted({start, (start + stop - 1) // 2, stop - 1}))
    return result


def world_metrics(sample, target):
    def mse(a, b):
        return float(F.mse_loss(a.float().cpu(), b.float().cpu()))
    reward = sample.reward_logits.float().cpu().reshape(-1)
    success = sample.success_logit.float().cpu().reshape(-1)
    reward_target = target["reward_chunk"].float().reshape(-1)
    valid = target["reward_chunk_mask"].bool().reshape(-1)
    return {
        "future_latent_mse": mse(sample.future[0], target["future_latents"]),
        "persistence_latent_mse": mse(target["current_latents"], target["future_latents"]),
        "future_state_mse": mse(sample.future_state[0, 0], target["future_state"]),
        "persistence_state_mse": mse(target["state"], target["future_state"]),
        "reward_bce": float(F.binary_cross_entropy_with_logits(reward[valid], reward_target[valid])),
        "reward_accuracy": float(((reward[valid] >= 0) == reward_target[valid].bool()).float().mean()),
        "success_bce": float(F.binary_cross_entropy_with_logits(success, target["success"].float().reshape(-1))),
        "success_probability": float(success.sigmoid().mean()),
        "success_target": float(target["success"].item()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True, help="Saved/config snapshot containing the four pools")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--render", action="store_true", help="Decode input/prediction/target VAE latents into an HTML report")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier probe.
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = restore_config_tuples(json.loads(Path(args.data_config).read_text()))
    from robonana.inference_contract import resolve_online_contract, read_contract
    overrides = dict.fromkeys(read_contract(args.checkpoint)["sampling"])
    contract = resolve_online_contract(args.checkpoint, config["models"]["checkpoint_dir"], overrides, inference_mode="action_only")
    settings = contract["sampling"]
    torch.manual_seed(args.seed)
    model, _ = load_flux2_fact_trained_checkpoint(args.checkpoint, config_path=args.model_config,
                      action_dim=config["models"]["action_dim"],
                      state_dim=config["models"]["state_dim"],
                      expert_hidden_dim=config["models"]["expert_hidden_dim"],
                      device=args.device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    vae = None
    if args.render:
        from diffusers.models import AutoencoderKLFlux2
        from robonana.training.visualization import decode_flux2_tokens
        from PIL import Image
        vae = AutoencoderKLFlux2.from_pretrained(config["models"]["checkpoint_dir"], subfolder="vae",
                  torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(args.device)
    schedule = flow_euler_schedule(settings["num_inference_steps"], flow_shift=settings["flow_shift"], device=args.device)
    rows = []
    for pool in configured_pools(config):
        cls = RoboTwinLeRobotDataset if pool["_class_name"] == "RoboTwinLeRobotDataset" else RoboTwinHDF5Dataset
        dataset = cls.load(pool)
        for index in probe_indices(dataset, args.episodes):
            item = dataset[index]
            # Reset noise by pool/index, independent of checkpoint load RNG use.
            torch.manual_seed(args.seed + int(item["pool_id"]) * 100000 + index)
            def batch(key):
                return item[key].unsqueeze(0).to(args.device, dtype=torch.bfloat16)
            current, state = batch("current_latents"), batch("state").unsqueeze(1)
            sampled = sample_mac_world(
                model=model, context=batch("context"), current_latents=current,
                state=state, context_mask=item["context_mask"].unsqueeze(0).to(args.device),
                clean_action=batch("behavior_action"), future_noise=torch.randn_like(current),
                future_state_noise=torch.randn_like(state),
                schedule=schedule,
                grid_height=12, grid_width=24,
            )
            metrics = world_metrics(sampled, item)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"nonfinite world probe: {pool['pool_name']} {index}")
            row = dict(pool=pool["pool_name"], index=index,
                       observation_id=item["observation_id"], **metrics)
            rows.append(row)
            if vae is not None:
                # 中文：逐图解码，限制叠加评估峰值；GT同样经过VAE，不能称为原始RGB。
                # English: Decode one image at a time; targets are VAE reconstructions, not raw RGB.
                panels = []
                for tokens in (current, sampled.future, batch("future_latents")):
                    pixels = decode_flux2_tokens(vae, tokens)[0].permute(1, 2, 0).cpu().numpy()
                    panels.append(Image.fromarray((pixels * 255).round().astype(np.uint8)))
                canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height))
                offset = 0
                for panel in panels:
                    canvas.paste(panel, (offset, 0))
                    offset += panel.width
                canvas.save(output / f"{pool['pool_name']}_{index}.png")
            torch.save(dict(predicted_latent=sampled.future.cpu(),
                            predicted_state=sampled.future_state.cpu(),
                            target_latent=item["future_latents"], target_state=item["future_state"],
                            reward_logits=sampled.reward_logits.cpu(), success_logit=sampled.success_logit.cpu()),
                       output / f"{pool['pool_name']}_{index}.pt")
            print(json.dumps(row), flush=True)
        dataset.close()
    summary = {
        pool: {key: float(np.mean([r[key] for r in rows if r["pool"] == pool]))
               for key in rows[0] if key not in {"pool", "index", "observation_id"}}
        for pool in sorted({r["pool"] for r in rows})
    }
    payload = dict(checkpoint=args.checkpoint, model_config=args.model_config,
                   data_config=args.data_config, seed=args.seed, sampling_steps=settings["num_inference_steps"],
                   evaluation_scope="training-set fit only; not held-out", rows=rows, summary=summary)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    if vae is not None:
        cards = []
        for row in rows:
            label = html.escape(str(row["observation_id"]))
            filename = f"{row['pool']}_{row['index']}.png"
            cards.append(f'<section><h3>{label}</h3><p>success: {row["success_probability"]:.2%}; GT: {row["success_target"]:.0f}; latent MSE: {row["future_latent_mse"]:.4f}; persistence: {row["persistence_latent_mse"]:.4f}</p><img src="{filename}" width="1152"></section>')
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>World model reconstruction</title><style>body{font-family:sans-serif;margin:24px;background:#eee}section{background:white;padding:16px;margin:16px 0}img{max-width:100%;height:auto}</style><h1>60k world-model sample report</h1><p>Training-set fit; real action; pure-noise future generation. Left: current VAE decode | middle: predicted t+48 | right: target VAE decode. Not raw RGB or policy success evaluation.</p>' + ''.join(cards), encoding="utf-8")


if __name__ == "__main__":
    main()
