from __future__ import annotations

import random

from robonana_robotwin_client import (
    _align_eval_instruction_with_training,
    _install_sampling_seed_hook,
    _seed_python_random,
    sampling_seed_for_step,
)


def test_eval_instruction_can_match_the_training_cache(monkeypatch) -> None:
    class FakeTask:
        instruction = "random paraphrase"

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

    task = FakeTask()
    monkeypatch.setenv(
        "ROBONANA_EVAL_INSTRUCTION",
        "Use the medium-sized metal hammer to hammer the block.",
    )

    assert _align_eval_instruction_with_training(task) == (
        "Use the medium-sized metal hammer to hammer the block."
    )
    assert task.get_instruction() == "Use the medium-sized metal hammer to hammer the block."


def test_dotted_robotwin_policy_adapter_exports_fact_hooks() -> None:
    from robonana_robotwin import adapter

    assert callable(adapter.get_model)
    assert callable(adapter.eval)
    assert callable(adapter.reset_model)


def test_small200m_robotwin_variant_matches_training_config() -> None:
    from robonana.inference import robotwin_model_params

    params = robotwin_model_params("small200m")

    assert params.hidden_size == 1024
    assert params.num_heads == 8
    assert params.depth == 2
    assert params.depth_single_blocks == 8
    assert params.context_in_dim == 7680


def test_sampling_seed_is_stable_per_replanning_point() -> None:
    assert sampling_seed_for_step(100013, 0, 48) == sampling_seed_for_step(100013, 47, 48)
    assert sampling_seed_for_step(100013, 48, 48) == sampling_seed_for_step(100013, 0, 48) + 1
    assert sampling_seed_for_step(100014, 0, 48) != sampling_seed_for_step(100013, 0, 48)


def test_fact_request_hook_forwards_sampling_seed() -> None:
    class FakeModel:
        def _build_request(self, example):
            return {"value": example["value"]}

    model = FakeModel()
    _install_sampling_seed_hook(model)
    model._robonana_sampling_seed = 1234

    assert model._build_request({"value": 7}) == {"value": 7, "sampling_seed": 1234}


def test_episode_seed_controls_python_random_instruction_selection() -> None:
    previous_state = random.getstate()
    try:
        _seed_python_random(100000)
        first = [random.random() for _ in range(4)]
        _seed_python_random(100000)
        second = [random.random() for _ in range(4)]
    finally:
        random.setstate(previous_state)

    assert first == second
