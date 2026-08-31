from robonana.configs.robotwin_flux2_4b_dino_reward_success_q_from150k import config


def test_reward_success_continuation_is_one_gpu_bs32_for_10k_steps():
    assert config["launch"]["gpu_ids"] == [6]
    assert config["dataloaders"]["train"]["batch_size_per_gpu"] == 32
    assert config["train"]["gradient_accumulation_steps"] == 1
    assert config["train"]["initial_global_step"] == 150000
    assert config["train"]["max_steps"] == 160000
    assert config["train"]["resume"] is False
    assert config["models"]["initialization"] == "trained"
    assert config["models"]["reward_head_type"] == "direct"
    assert config["models"]["success_dim"] == 1
    assert config["models"]["checkpoint"].endswith(
        "checkpoint_epoch_7_step_150000/transformer/diffusion_pytorch_model.bin"
    )
    assert config["train"]["loss_weights"]["reward_loss"] == 0.01
    assert config["train"]["loss_weights"]["success_loss"] == 0.01
    assert config["train"]["loss_weights"]["q_loss"] == 0.001
