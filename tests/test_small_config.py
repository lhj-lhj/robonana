from robonana.configs.robotwin_flux2_small200m import config


def test_small_config_preserves_data_contract_and_uses_bf16_ddp():
    data = config["dataloaders"]["train"]
    model = config["models"]
    train = config["train"]

    assert config["launch"]["distributed_type"] == "MULTI_GPU"
    assert "deepspeed_config" not in config["launch"]
    assert data["data_or_config"]["task_glob"] == "*/aloha-agilex_clean_50"
    assert data["sampler"] == {"type": "RoboTwinEpisodeSampler", "infinite": True}
    assert model["initialization"] == "scratch"
    assert model["params"]["context_in_dim"] == 7680
    assert model["value_dim"] == 1
    assert model["gradient_checkpointing"] is False
    assert train["mixed_precision"] == "bf16"
    assert train["activation_checkpointing"] is False
