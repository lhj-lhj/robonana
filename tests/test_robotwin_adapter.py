from __future__ import annotations

import io
import random
from types import SimpleNamespace

import numpy as np

import robonana_robotwin_client
from robonana_robotwin_client import (
    _ChunkValueOverlayStream,
    _align_eval_instruction_with_training,
    _install_chunk_value_hook,
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


def test_fact_response_hook_retains_one_value_for_the_action_chunk() -> None:
    class FakeClient:
        @staticmethod
        def inference(request):
            return {
                "action": np.zeros((3, 2), dtype=np.float32),
                "chunk_value": 0.625,
                "value_horizon": 24,
            }

    model = SimpleNamespace(client=FakeClient())
    _install_chunk_value_hook(model)

    model.client.inference({"observation": 1})
    assert model._robonana_chunk_value == 0.625
    assert model._robonana_value_horizon == 24
    assert model._robonana_chunk_index == 0


def test_video_overlay_reuses_chunk_value_for_every_raw_frame(monkeypatch) -> None:
    labels = []

    def fake_overlay(frame, label):
        labels.append(label)
        return frame

    monkeypatch.setattr(robonana_robotwin_client, "_overlay_value", fake_overlay)
    task = SimpleNamespace(
        now_obs={
            "observation": {
                "head_camera": {"rgb": np.zeros((3, 4, 3), dtype=np.uint8)}
            }
        }
    )
    model = SimpleNamespace(
        _robonana_chunk_value=0.75,
        _robonana_value_horizon=24,
        _robonana_chunk_index=2,
    )
    output = io.BytesIO()
    stream = _ChunkValueOverlayStream(output, task, model)
    frame = np.zeros((3, 4, 3), dtype=np.uint8).tobytes()

    stream.write(frame + frame)

    assert output.getvalue() == frame + frame
    assert labels == [
        "chunk=002  h=24  value=0.7500",
        "chunk=002  h=24  value=0.7500",
    ]


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
