import pytest

from robonana.configs.robotwin_flux2_4b_dino_reward_q_from120k import config
from robonana.training.robotwin_trainer import _validate_initial_global_step


def test_reward_q_continuation_uses_distinct_run_and_fresh_30k_schedule():
    assert config["models"]["initialization"] == "trained"
    assert config["models"]["checkpoint"].endswith(
        "checkpoint_epoch_6_step_120000/transformer/diffusion_pytorch_model.bin"
    )
    assert config["models"]["checkpoint_config"].endswith("120k/config.json")
    assert config["project_dir"] != str(
        config["models"]["checkpoint"].split("/models/", 1)[0]
    )
    assert config["train"]["initial_global_step"] == 120000
    assert config["train"]["max_steps"] == 150000
    assert config["train"]["resume"] is False
    assert config["schedulers"]["decay_steps"] == 30000
    assert config["train"]["loss_weights"]["reward_loss"] == 0.01
    assert config["train"]["loss_weights"]["q_loss"] == 0.001

    global_batch = (
        len(config["launch"]["gpu_ids"])
        * config["dataloaders"]["train"]["batch_size_per_gpu"]
        * config["train"]["gradient_accumulation_steps"]
    )
    assert global_batch == 256


def test_initial_global_step_validation():
    assert _validate_initial_global_step(120000, 150000) == 120000
    with pytest.raises(ValueError, match="cannot be negative"):
        _validate_initial_global_step(-1, 150000)
    with pytest.raises(ValueError, match="smaller than max_steps"):
        _validate_initial_global_step(150000, 150000)
