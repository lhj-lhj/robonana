from concurrent.futures import ThreadPoolExecutor
import pytest
import torch

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from robonana.inference.dynamic_batch_server import DynamicInferenceBatcher
from robonana.inference.robotwin_policy import InferenceMode
from world_action_model.pipeline.utils import NormalizationTensors


def test_dynamic_batcher_coalesces_concurrent_requests():
    class FakePolicy:
        def __init__(self):
            self.batch_sizes = []

        def inference_batch(self, observations):
            self.batch_sizes.append(len(observations))
            return [{"action": observation["id"]} for observation in observations]

    policy = FakePolicy()
    batcher = DynamicInferenceBatcher(policy, max_batch_size=3, max_wait_ms=100)
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(batcher.submit, {"id": index}) for index in range(3)]
            results = [future.result(timeout=2) for future in futures]
    finally:
        batcher.close()

    assert sorted(result["action"] for result in results) == [0, 1, 2]
    assert policy.batch_sizes == [3]


def test_dynamic_batcher_propagates_policy_errors_to_every_request():
    class FailingPolicy:
        def inference_batch(self, observations):
            raise RuntimeError("batch failed")

    batcher = DynamicInferenceBatcher(FailingPolicy(), max_batch_size=2, max_wait_ms=100)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(batcher.submit, {"id": index}) for index in range(2)]
            for future in futures:
                with pytest.raises(RuntimeError, match="batch failed"):
                    future.result(timeout=2)
    finally:
        batcher.close()


def test_batched_policy_returns_one_action_chunk_per_observation():
    policy = object.__new__(BatchedRoboNanaRobotWinPolicy)
    policy.inference_mode = InferenceMode.ACTION
    policy.return_stage2_image = False
    policy.model_device = torch.device("cpu")
    policy.vae_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.state_dim = 2
    policy.action_dim = 2
    policy.action_chunk = 3
    policy.delta_mask = torch.tensor([False, False])
    zeros = torch.zeros(2)
    ones = torch.ones(2)
    policy.normalization = NormalizationTensors(
        state_mean=zeros,
        state_std=ones,
        state_min=torch.full((2,), -2.0),
        state_max=torch.full((2,), 2.0),
        action_mean=zeros,
        action_std=ones,
        action_min=torch.full((2,), -1.0),
        action_max=torch.full((2,), 1.0),
        value_min=torch.tensor([-1.0]),
        value_max=torch.tensor([2.0]),
    )
    policy._sync = lambda device: None
    policy._batched_current_image_tokens = lambda observations: torch.zeros(
        len(observations), 2, 8
    )
    policy._batched_context = lambda observations: (
        torch.zeros(len(observations), 2, 4),
        torch.ones(len(observations), 2, dtype=torch.bool),
    )
    policy._sample_action_batch = lambda **kwargs: torch.zeros(2, 3, 2)

    responses = policy.inference_batch(
        [
            {"observation.state": torch.zeros(2), "instruction": "task a"},
            {"observation.state": torch.ones(2), "instruction": "task b"},
        ]
    )

    assert len(responses) == 2
    assert all(response["action"].shape == (3, 2) for response in responses)
    assert responses[0]["_policy_timing_ms"]["batch_size"] == 2
