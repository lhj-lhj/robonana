from pathlib import Path

import torch

import robonana.inference.batched_policy as batched_policy


def test_uncached_instructions_are_encoded_in_one_batch(monkeypatch) -> None:
    calls: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, checkpoint, device) -> None:
            assert checkpoint == Path("flux")
            assert torch.device(device) == torch.device("cpu")

        def __call__(self, instructions: list[str]) -> torch.Tensor:
            calls.append(list(instructions))
            values = [1.0 if instruction == "task a" else 2.0 for instruction in instructions]
            return torch.tensor(values).reshape(-1, 1, 1).expand(-1, 3, 4).clone()

    monkeypatch.setattr(batched_policy, "LocalQwen3Embedder", FakeEmbedder)
    policy = batched_policy.BatchedRoboNanaRobotWinPolicy.__new__(
        batched_policy.BatchedRoboNanaRobotWinPolicy
    )
    policy.flux_checkpoint_dir = Path("flux")
    policy.text_encoder_device = torch.device("cpu")
    policy.model_device = torch.device("cpu")
    policy.dtype = torch.float32
    policy._text_embedder = None
    policy._context_cache = {}
    observations = [
        {"instruction": "task a"},
        {"instruction": "task b"},
        {"instruction": "task a"},
    ]

    context, mask = policy._batched_context(observations)
    cached_context, cached_mask = policy._batched_context(observations)

    assert calls == [["task a", "task b"]]
    assert tuple(context.shape) == (3, 3, 4)
    assert torch.equal(context[:, 0, 0], torch.tensor([1.0, 2.0, 1.0]))
    assert mask.all()
    assert torch.equal(cached_context, context)
    assert torch.equal(cached_mask, mask)
