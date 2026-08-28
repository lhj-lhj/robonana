from concurrent.futures import ThreadPoolExecutor
import inspect
import threading

import pytest
import torch
from flux2.model import Flux2Params

from robonana.inference.batched_policy import BatchedRoboNanaRobotWinPolicy
from robonana.inference.dynamic_batch_server import (
    DynamicBatchRobotInferenceServer,
    DynamicInferenceBatcher,
)
from robonana.inference.robotwin_policy import InferenceMode
from robonana.models.flux2_fact import Flux2FACTModel
from robonana.sampling import flow_euler_schedule
from world_action_model.pipeline.utils import NormalizationTensors
from world_action_model.sockets import RobotInferenceClient


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


def test_fact_tcp_server_batches_two_persistent_clients():
    class FakePolicy:
        def __init__(self):
            self.batch_sizes = []

        def inference_batch(self, observations):
            self.batch_sizes.append(len(observations))
            return [{"action": observation["id"]} for observation in observations]

        def inference(self, observation):
            raise AssertionError("multi-client test must use inference_batch")

    policy = FakePolicy()
    server = DynamicBatchRobotInferenceServer(
        policy,
        host="127.0.0.1",
        port=0,
        max_batch_size=2,
        max_wait_ms=100,
        max_clients=4,
    )
    port = server.server_socket.getsockname()[1]
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    def infer(request_id):
        client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=2000)
        try:
            return client.inference({"id": request_id})
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(infer, index) for index in range(2)]
        results = [future.result(timeout=3) for future in futures]

    stop_client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=2000)
    try:
        stop_client.kill_server()
    finally:
        stop_client.close()
    server_thread.join(timeout=3)

    assert not server_thread.is_alive()
    assert sorted(result["action"] for result in results) == [0, 1]
    assert policy.batch_sizes == [2]


def test_batched_policy_returns_one_action_chunk_per_observation():
    policy = object.__new__(BatchedRoboNanaRobotWinPolicy)
    policy.inference_mode = InferenceMode.ACTION
    policy.return_chunk_q = False
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


def test_stage1_sampler_executes_one_true_model_batch():
    params = Flux2Params(
        in_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    policy = object.__new__(BatchedRoboNanaRobotWinPolicy)
    policy.model_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.horizon = 2
    policy.action_chunk = 3
    policy.grid_height = 1
    policy.grid_width = 2
    policy.state_dim = 3
    policy.action_dim = 4
    model_head_kwargs = (
        {"reward_dim": 1, "q_dim": 1}
        if "reward_dim" in inspect.signature(Flux2FACTModel).parameters
        else {"value_dim": 1}
    )
    policy.model = Flux2FACTModel(
        params,
        action_dim=4,
        state_dim=3,
        max_horizon=4,
        **model_head_kwargs,
    ).eval()
    policy.schedule = flow_euler_schedule(1, flow_shift=1.0, device="cpu")

    action = policy._sample_action_batch(
        context=torch.randn(2, 3, 16),
        context_mask=torch.tensor([[True, True, True], [True, True, False]]),
        current=torch.randn(2, 2, 8),
        state=torch.randn(2, 1, 3),
        sampling_seeds=[17, 23],
    )

    assert action.shape == (2, 3, 4)
    assert torch.isfinite(action).all()
