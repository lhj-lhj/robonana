import json
import ast
from pathlib import Path

import pytest

from robonana import inference_contract as contracts
from robonana.configs.posttrain_config import apply_mac_posttrain_config


def _posttrain(tmp_path):
    base = {"project_dir": str(tmp_path), "models": {},
            "dataloaders": {"train": {"data_or_config": {
                "_class_name": "RoboTwinLeRobotDataset", "data_path": str(tmp_path)}, "sampler": {}}},
            "train": {"loss_weights": {}, "tracker_init_kwargs": {"wandb": {}}}}
    return apply_mac_posttrain_config(base)["train"]["posttrain"]


def _saved(tmp_path):
    weights = tmp_path / "diffusion_pytorch_model.bin"
    weights.write_bytes(b"synthetic weights, not a real model")
    payload = dict(version=1, sampling=contracts.sampling_contract(_posttrain(tmp_path)),
                   imagination_candidate_count=8, image={"vae": "a"}, normalization_sha256="stats-a",
                   action_mapping="A_zscore_delta_to_absolute_no_clip_nonfinite_fallback_v1")
    contracts.write_contract(weights, payload, phase="world_policy", step=1)
    return weights, payload


def test_sampling_uses_saved_nondefault_values_and_separate_candidate_budgets(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBONANA_MAC_SAMPLING_STEPS", "7")
    monkeypatch.setenv("ROBONANA_MAC_FLOW_SHIFT", "2.5")
    monkeypatch.setenv("ROBONANA_MAC_TRAIN_CANDIDATES", "4")
    monkeypatch.setenv("ROBONANA_MAC_EVAL_CANDIDATES", "12")
    post = _posttrain(tmp_path)
    result = contracts.sampling_contract(post)
    assert result["num_inference_steps"] == 7
    assert result["flow_shift"] == 2.5
    assert result["rejection_candidate_count"] == 12
    assert post["imagination"]["candidate_count"] == 4


@pytest.mark.parametrize("field,value", [("sampling_steps", 0), ("flow_shift", float("nan")), ("flow_shift", -1)])
def test_invalid_sampling_rejected(tmp_path, field, value):
    post = _posttrain(tmp_path)
    post["imagination"][field] = value
    with pytest.raises(ValueError):
        contracts.sampling_contract(post)


def test_checkpoint_contract_roundtrip_and_weight_binding(tmp_path):
    weights, expected = _saved(tmp_path)
    contracts.check_contract(contracts.read_contract(weights), expected)
    weights.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="weight fingerprint mismatch"):
        contracts.read_contract(weights)


def test_old_checkpoint_never_infers_certificate_from_run_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"train": {"num_inference_steps": 20}}))
    with pytest.raises(FileNotFoundError, match="Uncertified checkpoint"):
        contracts.read_contract(tmp_path / "weights.bin")


@pytest.mark.parametrize("key", ["sampling", "image", "normalization_sha256", "imagination_candidate_count"])
def test_cross_stage_mismatch_rejected(tmp_path, key):
    weights, expected = _saved(tmp_path)
    expected[key] = "changed"
    with pytest.raises(ValueError, match=key):
        contracts.check_contract(contracts.read_contract(weights), expected)


def test_online_reads_settings_and_rejects_overrides_or_changed_inputs(monkeypatch, tmp_path):
    # No GPU/model loading: only input fingerprints and checkpoint metadata.
    import robonana.image_pipeline as images
    import robonana.normalization as normalization
    weights, expected = _saved(tmp_path)
    stats = tmp_path / "stats.json"
    stats.write_bytes(b"A")
    expected["normalization_sha256"] = contracts.sha256_file(stats)
    contracts.write_contract(weights, expected, phase="critic", step=10)
    monkeypatch.setattr(images, "image_contract", lambda path: {"vae": "a"})
    monkeypatch.setattr(normalization, "require_a_stats_path", lambda: stats)
    overrides = dict.fromkeys(expected["sampling"])
    actual = contracts.resolve_online_contract(weights, tmp_path, overrides)
    assert actual["sampling"] == expected["sampling"]
    for key, value in expected["sampling"].items():
        with pytest.raises(ValueError, match=key):
            contracts.resolve_online_contract(weights, tmp_path, dict(overrides, **{key: value + 1}))
    monkeypatch.setattr(images, "image_contract", lambda path: {"vae": "changed"})
    with pytest.raises(ValueError, match="image"):
        contracts.resolve_online_contract(weights, tmp_path, overrides)


def test_resume_checks_contract_before_fact_restores_weights(monkeypatch, tmp_path):
    from robonana.training.robotwin_trainer import RoboNanaTrainer
    from fact_train.trainers.trainer import Trainer
    trainer = object.__new__(RoboNanaTrainer)
    trainer.model_name = "transformer"
    trainer.inference_contract = {}
    directory = tmp_path / "transformer"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"old")
    monkeypatch.setattr(Trainer, "load_model_hook", lambda *args: pytest.fail("Must reject before restore"))
    with pytest.raises(FileNotFoundError, match="Uncertified"):
        trainer.load_model_hook([], str(tmp_path))


@pytest.mark.parametrize("phase", ["world_policy", "critic"])
def test_training_save_hook_binds_exported_weights(monkeypatch, tmp_path, phase):
    from types import SimpleNamespace
    from robonana.training.robotwin_trainer import RoboNanaTrainer
    from fact_train.trainers.trainer import Trainer
    _, expected = _saved(tmp_path)
    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer._cur_step = 10
    trainer._image_inputs_certified = True
    trainer.model_name = "transformer"
    trainer.mac_phase = phase
    trainer.inference_contract = expected
    trainer.target_value_ema = None
    def export(self, models, weights, output_dir):
        path = tmp_path / "export/transformer"
        path.mkdir(parents=True)
        (path / "diffusion_pytorch_model.bin").write_bytes(b"new trained weights")
    monkeypatch.setattr(Trainer, "save_model_hook", export)
    trainer.save_model_hook([], [], str(tmp_path / "export"))
    actual = contracts.read_contract(tmp_path / "export/transformer/diffusion_pytorch_model.bin")
    contracts.check_contract(actual, expected)
    assert actual["phase"] == phase
    assert actual["step"] == 10


def test_uncertified_training_cannot_publish_contract(monkeypatch, tmp_path):
    from robonana.training.robotwin_trainer import RoboNanaTrainer
    from fact_train.trainers.trainer import Trainer
    trainer = object.__new__(RoboNanaTrainer)
    trainer._cur_step = 1
    monkeypatch.setattr(Trainer, "save_model_hook", lambda *args: pytest.fail("Must not save"))
    with pytest.raises(RuntimeError, match="validated"):
        trainer.save_model_hook([], [], str(tmp_path))


@pytest.mark.parametrize("server", ["robotwin", "robotwin_batched", "robotwin_xpolicylab"])
def test_server_cli_has_no_independent_sampling_defaults(server):
    root = Path(__file__).resolve().parents[1]
    source = (root / f"scripts/services/inference_server_{server}.py").read_text(encoding="utf-8")
    names = {"--action-chunk", "--horizon", "--num-inference-steps", "--flow-shift",
             "--rejection-candidate-count", "--q-return-scale", "--discount",
             "--reward-non-goal", "--success-threshold"}
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            name = ast.literal_eval(node.args[0])
            if name in names:
                found.add(name)
                assert next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default") is None
    assert {"--num-inference-steps", "--flow-shift"} <= found
    assert "flow_shift=args.flow_shift" in source
    launcher = (root / "scripts/eval_robotwin_all_tasks_parallel.sh").read_text(encoding="utf-8")
    assert "--num-inference-steps 20" not in launcher
