import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from robonana.inference.selected_world import predict_selected_world
from robonana.inference.world_artifacts import save_selected_world


def test_world_uses_selected_action_and_private_rng(monkeypatch):
    captured = []
    def sampler(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            future=kwargs["future_noise"], future_state=kwargs["future_state_noise"],
            reward_logits=torch.zeros(1, 48, 1), success_logit=torch.tensor([[2.0]]),
        )
    monkeypatch.setattr("robonana.inference.selected_world.sample_mac_world", sampler)
    policy = SimpleNamespace(
        model=object(), schedule=torch.tensor([1., 0.]), grid_height=1, grid_width=1,
        action_chunk=48, discount=.999, reward_non_goal=-1., reward_goal=0.,
        _decode_stage2_image=lambda tokens: torch.zeros(1, 3, 1, 4, 6),
    )
    action = torch.ones(1, 48, 2)
    kwargs = dict(context=torch.zeros(1, 2, 4), context_mask=torch.ones(1, 2).bool(),
                  current=torch.zeros(1, 1, 8), state=torch.zeros(1, 1, 2),
                  clean_action=action, sampling_seed=123)
    rng = torch.get_rng_state().clone()
    result = predict_selected_world(policy, **kwargs)
    predict_selected_world(policy, **kwargs)
    assert captured[0]["clean_action"] is action
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(captured[0]["future_noise"], captured[1]["future_noise"])
    assert result["rewards"] == [-.5] * 48
    assert result["predicted_terminal"] and result["bootstrap_mask"] == 0
    expected = -.5 * sum(.999**i for i in range(48))
    assert abs(result["chunk_return"] - expected) < 1e-4


def test_artifact_retains_selected_action_reward_curve_and_image(tmp_path):
    world = dict(image=torch.zeros(1, 3, 1, 4, 6), rewards=[-1.] * 48,
                 success_probability=.2, predicted_terminal=False)
    response = dict(selected_world=world, selected_q=-212., selected_candidate_index=1,
                    candidate_q=torch.tensor([-300., -212.]), action=torch.ones(48, 2))
    request = {"observation.images.cam_high": np.ones((3, 4, 6), dtype=np.float32)}
    directory = save_selected_world(tmp_path, task="hanging_mug", seed=100000,
                                    step=48, request=request, response=response)
    record = json.loads((directory / "step_0048.json").read_text())
    assert record["selected_candidate_index"] == 1
    assert len(record["rewards"]) == 48 and len(record["action"]) == 48
    assert Image.open(directory / record["image"]).size == (6, 4)
    assert np.asarray(Image.open(directory / "step_0048_cam_high.png")).min() == 255
    assert "image" in response["selected_world"]
    action_only = dict(selected_world=world, action=torch.ones(48,2), _inference_mode='action_only')
    directory=save_selected_world(tmp_path,task='hanging_mug',seed=100001,step=0,
                                  request=request,response=action_only)
    record=json.loads((directory/'step_0000.json').read_text())
    assert record['selected_q'] is None and record['candidate_q']==[]
