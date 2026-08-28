from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

import robonana.training.posttraining as posttraining
from robonana.training.posttraining import (
    CandidateSearchResult,
    FullModelEMA,
    TDTargetResult,
    build_td_targets,
    search_failure_candidates,
)
import robonana.training.robotwin_trainer as trainer_module
from robonana.training.robotwin_trainer import RoboNanaTrainer


def _linear(weight: float) -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def test_full_model_ema_exact_init_and_polyak_update():
    online = _linear(1.0)
    ema = FullModelEMA(online, decay=0.995)
    torch.testing.assert_close(ema.model.weight, online.weight.float())
    assert not ema.model.training
    assert not any(parameter.requires_grad for parameter in ema.model.parameters())
    with torch.no_grad():
        online.weight.fill_(3.0)
    old = ema.model.weight.detach().clone()
    assert ema.update(online, optimizer_step=1, optimizer_step_succeeded=True)
    torch.testing.assert_close(
        ema.model.weight,
        old * 0.995 + torch.full_like(old, 3.0) * 0.005,
    )
    assert ema.update_count == 1


def test_ema_does_not_update_without_real_optimizer_step():
    online = _linear(1.0)
    ema = FullModelEMA(online, decay=0.995)
    before = ema.model.weight.detach().clone()
    with torch.no_grad():
        online.weight.fill_(9.0)
    assert not ema.update(online, optimizer_step=1, optimizer_step_succeeded=False)
    assert not ema.update(online, optimizer_step=1, optimizer_step_succeeded=False)
    torch.testing.assert_close(ema.model.weight, before)
    assert ema.update_count == 0


def test_rank_local_ema_updates_are_deterministic():
    online_a = _linear(2.0)
    online_b = _linear(2.0)
    ema_a = FullModelEMA(online_a, decay=0.995)
    ema_b = FullModelEMA(online_b, decay=0.995)
    with torch.no_grad():
        online_a.weight.fill_(4.0)
        online_b.weight.fill_(4.0)
    ema_a.update(online_a, optimizer_step=1, optimizer_step_succeeded=True)
    ema_b.update(online_b, optimizer_step=1, optimizer_step_succeeded=True)
    torch.testing.assert_close(ema_a.model.weight, ema_b.model.weight)


def test_ema_checkpoint_round_trip_and_legacy_exact_copy(tmp_path):
    online = _linear(1.0)
    ema = FullModelEMA(online, decay=0.995)
    with torch.no_grad():
        online.weight.fill_(5.0)
    ema.update(online, optimizer_step=1, optimizer_step_succeeded=True)
    path = tmp_path / "ema_model.safetensors"
    save_file(ema.state_dict(), str(path))
    restored = FullModelEMA(_linear(-1.0), decay=0.995)
    restored.load_state_dict(load_file(str(path)))
    torch.testing.assert_close(restored.model.weight, ema.model.weight)

    legacy = FullModelEMA(online, decay=0.995)
    torch.testing.assert_close(legacy.model.weight, online.weight.float())


def test_failure_candidate_search_is_online_best_of_8_and_ema_ranked(monkeypatch):
    online = _linear(1.0).train()
    ema = _linear(2.0).eval()
    calls = {"action_models": [], "q_models": [], "world_noise": []}

    def fake_actions(**kwargs):
        calls["action_models"].append(kwargs["model"])
        assert kwargs["model"].training is False
        return kwargs["action_noise"]

    def fake_q(**kwargs):
        calls["q_models"].append(kwargs["model"])
        calls["world_noise"].append(
            (
                kwargs["future_state_noise"].clone(),
                kwargs["reward_noise"].clone(),
                kwargs["q_noise"].clone(),
                kwargs["horizon"],
            )
        )
        return kwargs["clean_action"][:, 0, 0].float()

    monkeypatch.setattr(posttraining, "_sample_actions_microbatched", fake_actions)
    monkeypatch.setattr(posttraining, "_sample_q_microbatched", fake_q)
    batch, candidates, horizon, action_dim = 2, 8, 48, 2
    values = torch.arange(candidates, dtype=torch.float32).view(1, candidates, 1, 1)
    action_noise = values.expand(batch, candidates, horizon, action_dim).clone()
    behavior = torch.full((batch, horizon, action_dim), -10.0)
    result = search_failure_candidates(
        online_model=online,
        ema_model=ema,
        context=torch.zeros(batch, 3, 4),
        current_latents=torch.zeros(batch, 2, 8),
        state=torch.zeros(batch, 1, action_dim),
        context_mask=torch.ones(batch, 3, dtype=torch.bool),
        behavior_action=behavior,
        action_noise=action_noise,
        action_sampling_steps=1,
        q_sampling_steps=1,
        microbatch_size=16,
    )
    assert online.training is True
    assert calls["action_models"] == [online]
    assert calls["q_models"] == [ema, ema]
    assert result.candidate_q.shape == (batch, 8)
    assert result.best_index.tolist() == [7, 7]
    torch.testing.assert_close(result.pseudo_action, torch.full_like(behavior, 7.0))
    assert not torch.equal(result.pseudo_action, behavior)
    assert result.behavior_q.tolist() == [-10.0, -10.0]
    candidate_state_noise, candidate_reward_noise, candidate_q_noise, used_horizon = calls[
        "world_noise"
    ][0]
    assert used_horizon == 48
    for noise in (candidate_state_noise, candidate_reward_noise, candidate_q_noise):
        grouped = noise.reshape(batch, candidates, *noise.shape[1:])
        torch.testing.assert_close(grouped, grouped[:, :1].expand_as(grouped))
    assert not torch.equal(action_noise[:, 0], action_noise[:, 1])


def test_td_uses_ema_and_failure_timeout_bootstraps_but_delta_zero_is_masked(monkeypatch):
    ema = _linear(2.0).eval()
    captured = {"action_models": [], "q_models": [], "next_latents": []}

    def fake_actions(**kwargs):
        captured["action_models"].append(kwargs["model"])
        captured["next_latents"].append(kwargs["current_latents"].clone())
        return torch.ones_like(kwargs["action_noise"])

    def fake_q(**kwargs):
        captured["q_models"].append(kwargs["model"])
        return torch.full((kwargs["clean_action"].shape[0],), 5.0)

    monkeypatch.setattr(posttraining, "_sample_actions_microbatched", fake_actions)
    monkeypatch.setattr(posttraining, "_sample_q_microbatched", fake_q)
    batch = 4
    next_latents = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1).expand(batch, 2, 8)
    result = build_td_targets(
        ema_model=ema,
        context=torch.zeros(batch, 3, 4),
        next_current_latents=next_latents,
        next_state=torch.zeros(batch, 1, 2),
        context_mask=torch.ones(batch, 3, dtype=torch.bool),
        reward_h=torch.tensor([-1.0, -2.0, -3.0, 0.0]),
        delta_steps=torch.tensor([2, 3, 4, 0]),
        success_terminal_h=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        action_template=torch.zeros(batch, 48, 2),
        action_sampling_steps=1,
        q_sampling_steps=1,
    )
    assert captured["action_models"] == [ema]
    assert captured["q_models"] == [ema]
    # Only rows 1 and 2 bootstrap: row 0 is success terminal; row 3 has delta=0.
    assert captured["next_latents"][0][:, 0, 0].tolist() == [1.0, 2.0]
    assert result.bootstrap_mask.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert result.q_loss_mask.tolist() == [1.0, 1.0, 1.0, 0.0]
    expected = torch.tensor(
        [
            -1.0,
            -2.0 + 0.999**3 * 5.0,
            -3.0 + 0.999**4 * 5.0,
            0.0,
        ]
    )
    torch.testing.assert_close(result.q_target.reshape(-1), expected)
    assert not result.q_target.requires_grad
    assert not result.next_action.requires_grad


def test_trainer_searches_both_failure_pools_and_keeps_behavior_world_condition(monkeypatch):
    trainer = object.__new__(RoboNanaTrainer)
    trainer.full_model_ema = SimpleNamespace(model=torch.nn.Identity())
    trainer._models = [torch.nn.Identity()]
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.mixed_precision = "no"
    trainer.grid_height = 1
    trainer.grid_width = 1
    trainer.flow_shift = 1.0
    trainer.ema_forward_autocast_dtype = torch.bfloat16
    trainer._posttrain_metrics = {}
    trainer._last_candidate_search = None
    trainer._last_td_target = None
    trainer._last_failure_timeout_observation_ids = []
    trainer.posttrain_config = {
        "discount": 0.999,
        "failure_policy_improvement": {
            "candidate_count": 8,
            "candidate_horizon": 48,
            "candidate_action_sampling_steps": 20,
            "candidate_q_sampling_steps": 20,
            "flow_shift": 1.0,
            "candidate_microbatch_size": 16,
        },
        "td": {
            "target_action_horizon": 48,
            "action_sampling_steps": 20,
            "q_sampling_steps": 20,
            "flow_shift": 1.0,
            "microbatch_size": 16,
        },
    }
    behavior = torch.arange(4 * 48 * 2, dtype=torch.float32).reshape(4, 48, 2)
    pseudo = torch.full((2, 48, 2), 123.0)
    seen = {}

    def fake_search(**kwargs):
        seen["failure_behavior"] = kwargs["behavior_action"].clone()
        return CandidateSearchResult(
            pseudo_action=pseudo,
            candidates=pseudo[:, None].expand(-1, 8, -1, -1),
            candidate_q=torch.arange(8, dtype=torch.float32)[None].expand(2, -1),
            best_index=torch.tensor([7, 7]),
            best_q=torch.tensor([7.0, 7.0]),
            behavior_q=torch.tensor([1.0, 2.0]),
            elapsed_ms=1.0,
            peak_memory_bytes=0,
        )

    def fake_td(**kwargs):
        seen["td_action_template"] = kwargs["action_template"].clone()
        return TDTargetResult(
            q_target=torch.arange(4, dtype=torch.float32).reshape(4, 1, 1),
            next_q=torch.ones(4, 1, 1),
            next_action=torch.zeros_like(behavior),
            bootstrap_mask=torch.ones(4),
            q_loss_mask=torch.ones(4),
            discount_factor=torch.full((4,), 0.999),
            elapsed_ms=2.0,
            peak_memory_bytes=0,
        )

    monkeypatch.setattr(trainer_module, "search_failure_candidates", fake_search)
    monkeypatch.setattr(trainer_module, "build_td_targets", fake_td)
    batch = {
        "failure_episode_mask": torch.tensor([0, 0, 1, 1]),
        "pool_id": torch.tensor([0, 1, 2, 3]),
        "delta_steps": torch.ones(4, dtype=torch.long),
        "success_terminal_h": torch.zeros(4),
        "time_limit_truncated_h": torch.tensor([0, 0, 0, 1]),
        "observation_id": ["original", "collected", "historical", "latest-final"],
    }
    target, q, action_mask, q_mask = trainer._prepare_posttrain_targets(
        batch_dict=batch,
        context=torch.zeros(4, 1, 3),
        context_mask=torch.ones(4, 1, dtype=torch.bool),
        current=torch.zeros(4, 1, 8),
        future=torch.zeros(4, 1, 8),
        state=torch.zeros(4, 1, 2),
        future_state=torch.zeros(4, 1, 2),
        behavior_action=behavior,
        reward=torch.full((4, 1, 1), -1.0),
    )

    torch.testing.assert_close(target[:2], behavior[:2])
    torch.testing.assert_close(target[2:], pseudo)
    torch.testing.assert_close(seen["failure_behavior"], behavior[2:])
    torch.testing.assert_close(seen["td_action_template"], behavior)
    assert action_mask.tolist() == [1, 1, 1, 1]
    assert q_mask.tolist() == [1, 1, 1, 1]
    assert q.reshape(-1).tolist() == [0, 1, 2, 3]
    assert trainer._last_failure_timeout_observation_ids == ["latest-final"]


def _checkpoint_hook_trainer(model: torch.nn.Module):
    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        unwrap_model=lambda value, **kwargs: value,
        get_state_dict=lambda value: value.state_dict(),
    )
    trainer._models = [model]
    trainer._optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)]
    trainer.full_model_ema = FullModelEMA(model, decay=0.995)
    trainer.model_name = "transformer"
    trainer.checkpoint_safe_serialization = True
    trainer.checkpoint_strict = True
    trainer.with_ema = False
    trainer.current_collection_round = 4
    trainer.posttrain_config = {"enabled": True, "ema": {"decay": 0.995}}
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    return trainer


def test_trainer_checkpoint_saves_and_restores_full_ema_and_legacy_fallback(tmp_path):
    output = tmp_path / "checkpoint"
    output.mkdir()
    online = _linear(3.0)
    trainer = _checkpoint_hook_trainer(online)
    with torch.no_grad():
        trainer.full_model_ema.model.weight.fill_(2.0)
    trainer.full_model_ema.update_count = 17
    trainer.save_model_hook([online], [online.state_dict()], str(output))

    assert (output / "ema_model.safetensors").is_file()
    ema_state = json.loads((output / "ema_state.json").read_text())
    assert ema_state["update_count"] == 17
    assert ema_state["current_collection_round"] == 4
    assert json.loads((output / "posttrain_config.json").read_text())["enabled"] is True

    restored_online = _linear(-1.0)
    restored = _checkpoint_hook_trainer(restored_online)
    models = [restored_online]
    restored.load_model_hook(models, str(output))
    assert models == []
    torch.testing.assert_close(restored_online.weight, torch.full_like(restored_online.weight, 3.0))
    torch.testing.assert_close(
        restored.full_model_ema.model.weight,
        torch.full_like(restored.full_model_ema.model.weight, 2.0),
    )
    assert restored.full_model_ema.update_count == 17
    assert restored.current_collection_round == 4

    (output / "ema_model.safetensors").unlink()
    (output / "ema_state.json").unlink()
    legacy_online = _linear(-9.0)
    legacy = _checkpoint_hook_trainer(legacy_online)
    legacy.load_model_hook([legacy_online], str(output))
    torch.testing.assert_close(legacy_online.weight, torch.full_like(legacy_online.weight, 3.0))
    torch.testing.assert_close(legacy.full_model_ema.model.weight, legacy_online.weight.float())
    assert legacy.full_model_ema.update_count == 0


def test_forward_uses_pseudo_only_for_pred_action_and_behavior_for_clean_world(monkeypatch):
    captured = {}

    class CaptureModel(torch.nn.Module):
        def forward(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                image=kwargs["noisy_future_latents"],
                action=kwargs["noisy_pred_action"],
                future_state=kwargs["noisy_future_state"],
                reward=kwargs["noisy_reward"],
                q=kwargs["noisy_q"],
                dino=None,
            )

    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.mixed_precision = "no"
    trainer._models = [CaptureModel()]
    trainer.dino_dim = None
    trainer.dino_encoder = None
    trainer.grid_height = 1
    trainer.grid_width = 1
    trainer.flow_shift = 1.0
    trainer.pixel_eval_interval = 0
    trainer._cur_step = 1
    trainer.posttrain_enabled = True
    trainer._posttrain_metrics = {}
    behavior = torch.arange(2 * 48 * 2, dtype=torch.float32).reshape(2, 48, 2)
    pred_target = behavior.clone()
    pred_target[1].fill_(77.0)
    clean_q = torch.tensor([[[3.0]], [[4.0]]])
    trainer._prepare_posttrain_targets = lambda **kwargs: (
        pred_target,
        clean_q,
        torch.ones(2),
        torch.ones(2),
    )
    monkeypatch.setattr(
        trainer_module,
        "flow_noise",
        lambda clean, timestep: (clean, clean),
    )
    batch = {
        "context": torch.zeros(2, 1, 3),
        "context_mask": torch.ones(2, 1, dtype=torch.bool),
        "current_latents": torch.zeros(2, 1, 8),
        "future_latents": torch.ones(2, 1, 8),
        "state": torch.zeros(2, 2),
        "future_state": torch.ones(2, 2),
        "action": behavior,
        "behavior_action": behavior,
        "reward": torch.tensor([[-1.0], [-2.0]]),
        "q": torch.zeros(2, 1),
        "horizon_idx": torch.tensor([12, 48]),
        "action_loss_mask": torch.ones(2),
        "q_loss_mask": torch.ones(2),
        "failure_episode_mask": torch.tensor([0.0, 1.0]),
    }

    trainer.forward_step(batch)

    torch.testing.assert_close(captured["noisy_pred_action"], pred_target)
    torch.testing.assert_close(captured["gt_action_cond"], behavior)
    torch.testing.assert_close(captured["noisy_q"], clean_q)
    # There is no second action condition in the world path: S/R/Q/I/D can
    # only read the behavior-backed clean G track through the attention mask.
    assert "pseudo_action" not in captured
