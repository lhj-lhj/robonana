"""Guard the single Stage-1 normalization across data, configs and inference."""
import json
import pytest
from robonana import normalization


def test_only_a_is_accepted_even_when_another_file_is_equal(tmp_path, monkeypatch):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    payload = {"norm_stats": {"observation.state": {"mean": [0], "std": [1]}}}
    a.write_text(json.dumps(payload))
    b.write_text(a.read_text())
    monkeypatch.setattr(normalization, "A_STATS_PATH", a)
    assert normalization.load_a_stats() == payload
    assert normalization.load_a_stats(a) == payload
    with pytest.raises(ValueError, match="Only Stage-1"):
        normalization.load_a_stats(b)


def test_missing_a_never_falls_back_to_other_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(normalization, "A_STATS_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        normalization.load_a_stats()


def test_policy_rejects_b_before_loading_models(monkeypatch):
    pytest.importorskip("torch")  # Full policy import is validated on 190.
    from robonana.inference.robotwin_policy import RoboNanaRobotWinPolicy
    with pytest.raises(ValueError, match="Only Stage-1"):
        RoboNanaRobotWinPolicy(checkpoint="unused", flux_checkpoint_dir="unused",
                              stats_path="/old/B.json")
