import json

import pytest
import torch

from robonana.data.flux_cache import canonical_instruction, select_current_future_latents


def test_select_current_future_latents_uses_horizon_and_clamps():
    latents = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    current, future = select_current_future_latents(latents, current_index=3, horizon_idx=4)
    torch.testing.assert_close(current, latents[3])
    torch.testing.assert_close(future, latents[4])


def test_select_current_future_latents_rejects_nonpositive_horizon():
    with pytest.raises(ValueError, match="positive"):
        select_current_future_latents(torch.zeros(2, 3, 4), current_index=0, horizon_idx=0)


def test_canonical_instruction_prefers_first_seen_prompt(tmp_path):
    task_dir = tmp_path / "adjust_bottle" / "aloha-agilex_clean_50"
    instruction_dir = task_dir / "instructions"
    instruction_dir.mkdir(parents=True)
    path = instruction_dir / "episode0.json"
    path.write_text(json.dumps({"seen": ["  lift the bottle  "], "unseen": ["fallback"]}), encoding="utf-8")

    prompt, source = canonical_instruction(task_dir)

    assert prompt == "lift the bottle"
    assert source == path
