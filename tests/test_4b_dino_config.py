from accelerate import init_empty_weights
from flux2.model import Flux2Params

from robonana.configs.robotwin_flux2_4b_dino import config
from robonana.models.flux2_fact import Flux2FACTModel


def test_4b_dino_training_contract():
    assert config["launch"]["distributed_type"] == "DEEPSPEED"
    assert config["dataloaders"]["train"]["batch_size_per_gpu"] == 16
    assert config["train"]["gradient_accumulation_steps"] == 2
    assert config["train"]["pixel_eval_interval"] == 2000
    assert config["models"]["gradient_checkpointing"] is False
    assert config["models"]["initialization"] == "pretrained"
    assert config["models"]["dino_dim"] == 3072
    assert config["models"]["params"]["hidden_size"] == 3072
    assert config["dataloaders"]["train"]["data_or_config"]["dino_online"] is True
    assert config["optimizers"]["lr"] == 2e-5
    assert config["optimizers"]["robot_lr"] == 1e-4


def test_4b_dino_adapters_match_backbone_hidden_size():
    params = Flux2Params(**config["models"]["params"])
    with init_empty_weights():
        model = Flux2FACTModel(
            params,
            action_dim=14,
            state_dim=14,
            max_horizon=48,
            dino_dim=3072,
        )
    assert model.dino_in.in_features == 3072
    assert model.dino_in.out_features == 3072
    assert model.dino_out.in_features == 3072
    assert model.dino_out.out_features == 3072
