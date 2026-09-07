"""Opt-in diagnostics for the exact action selected by MAC Q argmax."""

from __future__ import annotations

import torch

from robonana.sampling import sample_mac_world


@torch.inference_mode()
def predict_selected_world(policy, *, context, context_mask, current, state,
                           clean_action, sampling_seed):
    """Reuse the training rollout sampler; never resample or rerank the action.

    MAC reference: https://github.com/kwanyoungpark/MAC
    This is the same one-transition world sampler used by our critic training,
    with an independent RNG so enabling diagnostics cannot advance policy RNG.
    The image is the fixed t+48 endpoint, not a generated 48-frame video.
    """
    world_seed = (int(sampling_seed or 0) + 2_000_000_011) % (2**63 - 1)
    generator = torch.Generator(device=current.device).manual_seed(world_seed)
    world = sample_mac_world(
        model=policy.model, context=context, context_mask=context_mask,
        current_latents=current, state=state, clean_action=clean_action,
        future_noise=torch.randn(current.shape, device=current.device,
                                 dtype=current.dtype, generator=generator),
        future_state_noise=torch.randn(state.shape, device=state.device,
                                       dtype=state.dtype, generator=generator),
        schedule=policy.schedule, grid_height=policy.grid_height,
        grid_width=policy.grid_width,
    )
    logits = world.reward_logits[0].float().reshape(-1)
    if logits.numel() != policy.action_chunk:
        raise ValueError(f"expected {policy.action_chunk} rewards, got {logits.numel()}")
    probabilities = logits.sigmoid()
    rewards = policy.reward_non_goal + probabilities * (
        policy.reward_goal - policy.reward_non_goal
    )
    success_logit = float(world.success_logit[0].float().reshape(-1)[0].item())
    success_probability = float(torch.sigmoid(torch.tensor(success_logit)).item())
    discounts = policy.discount ** torch.arange(rewards.numel(), device=rewards.device)
    image = policy._decode_stage2_image(world.future).detach().cpu()
    return {
        "image": image,
        "horizon": policy.action_chunk,
        "world_seed": world_seed,
        "reward_logits": logits.cpu().tolist(),
        "reward_probabilities": probabilities.cpu().tolist(),
        # Match imaginary critic training: expected rewards, not thresholded ones.
        "rewards": rewards.cpu().tolist(),
        "chunk_return": float((rewards * discounts).sum().item()),
        "discount": policy.discount,
        "success_logit": success_logit,
        "success_probability": success_probability,
        "predicted_terminal": success_probability >= 0.5,
        "bootstrap_mask": int(success_probability < 0.5),
        "clean_action_normalized": clean_action[0].float().cpu().tolist(),
        "future_state_normalized": world.future_state[0].float().cpu().tolist(),
    }
