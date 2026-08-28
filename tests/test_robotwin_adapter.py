from __future__ import annotations

import io
import random
from types import SimpleNamespace

import numpy as np

import robonana_robotwin_client
from robonana_robotwin_client import (
    _ChunkReturnOverlayStream,
    _align_eval_instruction_with_training,
    _install_chunk_return_hook,
    _install_sampling_seed_hook,
    _save_pending_stage2_image,
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


def test_fact_response_hook_retains_reward_and_q_for_the_action_chunk() -> None:
    class FakeClient:
        @staticmethod
        def inference(request):
            return {
                "action": np.zeros((3, 2), dtype=np.float32),
                "chunk_reward": -3.0,
                "chunk_q": -7.0,
                "return_horizon": 24,
                "images": np.zeros((1, 3, 1, 4, 8), dtype=np.float32),
            }

    model = SimpleNamespace(client=FakeClient())
    _install_chunk_return_hook(model)

    model.client.inference({"observation": 1})
    assert model._robonana_chunk_reward == -3.0
    assert model._robonana_chunk_q == -7.0
    assert model._robonana_return_horizon == 24
    assert model._robonana_chunk_index == 0
    assert model._robonana_pending_stage2_image.shape == (1, 3, 1, 4, 8)


def test_stage2_image_is_saved_once_per_new_chunk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROBONANA_STAGE2_IMAGE_ROOT", str(tmp_path))
    task = SimpleNamespace(task_name="beat_block_hammer", test_num=7)
    model = SimpleNamespace(
        _robonana_pending_stage2_image=np.zeros((1, 3, 1, 4, 8), dtype=np.float32),
        _robonana_chunk_index=2,
        _robonana_return_horizon=24,
    )

    output = _save_pending_stage2_image(task, model)

    assert output == tmp_path / "beat_block_hammer" / "episode_000007" / "chunk_002_h_024.png"
    assert output.is_file()
    assert _save_pending_stage2_image(task, model) is None


def test_video_overlay_reuses_chunk_return_for_every_raw_frame(monkeypatch) -> None:
    labels = []

    def fake_overlay(frame, label):
        labels.append(label)
        return frame

    monkeypatch.setattr(robonana_robotwin_client, "_overlay_return", fake_overlay)
    task = SimpleNamespace(
        now_obs={
            "observation": {
                "head_camera": {"rgb": np.zeros((3, 4, 3), dtype=np.uint8)}
            }
        }
    )
    model = SimpleNamespace(
        _robonana_chunk_reward=-3.0,
        _robonana_chunk_q=-7.0,
        _robonana_return_horizon=24,
        _robonana_chunk_index=2,
    )
    output = io.BytesIO()
    stream = _ChunkReturnOverlayStream(output, task, model)
    frame = np.zeros((3, 4, 3), dtype=np.uint8).tobytes()

    stream.write(frame + frame)

    assert output.getvalue() == frame + frame
    assert labels == [
        "chunk=002  h=24  reward=-3.0000  Q=-7.0000",
        "chunk=002  h=24  reward=-3.0000  Q=-7.0000",
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
