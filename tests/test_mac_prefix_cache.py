"""Full asymmetric FLUX forwards are the oracle for cached MAC execution."""

from unittest.mock import patch
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flux2.model import Flux2Params

from robonana.models.mac_flux2_fact import MacFlux2FACTModel
from robonana.models.position_ids import image_position_ids, text_position_ids
from robonana.sampling import (
    evaluate_mac_critics, flow_euler_step, sample_flux2_action, sample_q_rejection,
)


def model_and_inputs():
    torch.manual_seed(42)
    model = MacFlux2FACTModel(Flux2Params(
        in_channels=8, context_in_dim=16, hidden_size=32, num_heads=4,
        depth=2, depth_single_blocks=2, axes_dim=[2, 2, 2, 2],
        mlp_ratio=2.0, use_guidance_embed=False,
    ), action_dim=6, state_dim=6, expert_hidden_dim=16).eval()
    # A mixed batch with padding catches incorrect B/M indexing and key masks.
    inputs = dict(
        context=torch.randn(2, 3, 16), context_ids=text_position_ids(2, 3, "cpu"),
        current_latents=torch.randn(2, 2, 8),
        current_ids=image_position_ids(2, grid_height=1, grid_width=2,
                                      time_coord=torch.zeros(2, dtype=torch.long), device="cpu"),
        state=torch.randn(2, 1, 6),
        context_mask=torch.tensor([[True, True, False], [False, False, False]]),
    )
    return model, inputs


def sampling_inputs(inputs):
    return {key: value for key, value in inputs.items() if key not in ("context_ids", "current_ids")}


def test_model_disables_bf16_reduced_precision_reduction():
    # Model construction is shared by training, checkpoint loading, and
    # inference, including processes launched independently by accelerate.
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        model_and_inputs()
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def full_action(model, inputs, action, sigma):
    batch = action.shape[0]
    return model(
        **inputs, noisy_future_latents=action.new_empty(batch, 0, 8),
        future_ids=torch.empty(batch, 0, 4, dtype=torch.long),
        noisy_pred_action=action, gt_action_cond=action[:, :0],
        chunk_horizon=torch.full((batch,), 48),
        noisy_future_state=action.new_empty(batch, 0, 6),
        noisy_reward=action.new_empty(batch, 0, 1), noisy_q=action.new_empty(batch, 0, 1),
        action_timestep=sigma.expand(batch), wm_timestep=torch.zeros(batch),
    ).action


@pytest.mark.parametrize("count,group_size", [(1, 1), (5, 2), (5, 8)])
def test_cached_q_matches_full_forward_and_isolates_candidates(count, group_size):
    model, inputs = model_and_inputs()
    actions = torch.randn(2, count, 48, 6)
    cache = model.prefill_condition_cache(**inputs)
    original = [{name: value.clone() for name, value in layer.items()}
                for layer in (*cache.double, *cache.single)]
    with torch.no_grad():
        actual = model.score_q_candidates(cache, actions, candidate_batch_size=group_size)
        expected = []
        for index in range(count):
            full = model.prefill_critic_cache(**inputs, clean_action=actions[:, index])
            for shared, all_tokens in zip((*cache.double, *cache.single), (*full.double, *full.single)):
                for name in ("k", "v"):
                    torch.testing.assert_close(shared[name], all_tokens[name][:, :cache.prefix_length])
            pe = model._expert_query_pe(batch=2, device=torch.device("cpu"), dtype=torch.long, segment_id=11)
            expected.append(model.q_expert(full, query_pe=pe))
        torch.testing.assert_close(actual, torch.cat(expected, dim=1), atol=2e-6, rtol=2e-5)
        altered = actions.clone()
        altered[0, 0] += 3
        new_q = model.score_q_candidates(cache, altered, candidate_batch_size=group_size)
        torch.testing.assert_close(new_q[1], actual[1])
        torch.testing.assert_close(new_q[0, 1:], actual[0, 1:])
        assert not torch.isclose(new_q[0, 0], actual[0, 0])
    for before, after in zip(original, (*cache.double, *cache.single)):
        for name in before:
            assert torch.equal(before[name], after[name])
            assert not after[name].requires_grad


def test_cached_actor_matches_full_flow_and_rejection_prefills_once():
    model, inputs = model_and_inputs()
    noise = torch.randn(2, 5, 48, 6)
    schedule = torch.tensor([1.0, 0.4, 0.0])
    with torch.no_grad():
        expected = []
        for candidate in range(5):
            action = noise[:, candidate].clone()
            for sigma, sigma_next in zip(schedule[:-1], schedule[1:]):
                action = flow_euler_step(action, full_action(model, inputs, action, sigma), sigma, sigma_next)
            expected.append(action)
        expected = torch.stack(expected, dim=1)
    with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill, \
         patch.object(model.value_expert, "forward", side_effect=AssertionError("rejection called Value")):
        result = sample_q_rejection(
            model=model, **sampling_inputs(inputs), action_noise=noise, schedule=schedule,
            candidate_count=5, candidate_batch_size=2, grid_height=1, grid_width=2,
        )
        assert prefill.call_count == 1
        assert prefill.call_args.kwargs["context"].shape[0] == 2
    torch.testing.assert_close(result.candidates, expected, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(result.best_index, result.candidate_q.argmax(dim=1))
    with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
        single = sample_flux2_action(
            model=model, **sampling_inputs(inputs), action_noise=noise[:, 0],
            schedule=schedule, chunk_horizon=48, grid_height=1, grid_width=2,
        )
        assert prefill.call_count == 1
    torch.testing.assert_close(single, expected[:, 0], atol=3e-6, rtol=2e-5)


def test_critic_joint_forward_shares_c_and_preserves_expert_gradients():
    model, inputs = model_and_inputs()
    model.set_training_phase("critic")
    action = torch.randn(2, 48, 6)
    with patch.object(model, "prefill_condition_cache", wraps=model.prefill_condition_cache) as prefill:
        value, q = evaluate_mac_critics(
            model=model, **sampling_inputs(inputs), clean_action=action, grid_height=1, grid_width=2,
        )
        assert prefill.call_count == 1
    (value.square().mean() + q.square().mean()).backward()
    cached_grads = {name: p.grad.clone() for name, p in model.q_expert.named_parameters()}
    assert all(p.grad is not None for p in model.value_expert.parameters())
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith(("value_expert.", "q_expert.")))
    model.zero_grad(set_to_none=True)
    full = model.prefill_critic_cache(**inputs, clean_action=action)
    pe = model._expert_query_pe(batch=2, device=torch.device("cpu"), dtype=torch.long, segment_id=11)
    model.q_expert(full, query_pe=pe).square().mean().backward()
    for name, p in model.q_expert.named_parameters():
        torch.testing.assert_close(p.grad, cached_grads[name], atol=3e-6, rtol=3e-5)


def _ddp_critic_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        model, inputs = model_and_inputs()
        model.set_training_phase("critic")
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        inputs["state"] += rank * 0.1
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            cache = model.prefill_condition_cache(**inputs)
            value, q = evaluate_mac_critics(
                model=wrapped, **sampling_inputs(inputs), clean_action=torch.randn(2, 48, 6),
                grid_height=1, grid_width=2,
                condition_cache=cache,
            )
            (value.square().mean() + q.square().mean()).backward()
            for expert in (model.value_expert, model.q_expert):
                grad = expert.head.linear.weight.grad
                gathered = [torch.empty_like(grad) for _ in range(2)]
                dist.all_gather(gathered, grad)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            assert all(p.grad is None for name, p in model.named_parameters()
                       if not name.startswith(("value_expert.", "q_expert.")))
            optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for two-rank critic validation")
def test_two_rank_critic_runs_two_steps_with_shared_frozen_prefix(tmp_path):
    mp.spawn(_ddp_critic_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
