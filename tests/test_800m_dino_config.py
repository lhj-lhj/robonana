from accelerate import init_empty_weights
from flux2.model import Flux2Params

from robonana.configs.robotwin_flux2_800m_dino import config
from robonana.models.flux2_fact import Flux2FACTModel


def test_800m_dino_config_enables_strict_online_encoder_contract():
    data = config["dataloaders"]["train"]["data_or_config"]
    assert config["models"]["dino_dim"] == 3072
    assert config["models"]["dino_encoder_model"] == "vit_base_patch16_dinov3.lvd1689m"
    assert config["models"]["dino_encoder_batch_size"] == 96
    assert data["dino_online"] is True
    assert "dino_cache" not in data
    assert config["train"]["loss_weights"]["dino_loss"] == 0.1


def test_800m_dino_heads_add_expected_parameters():
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
    assert model.dino_in.out_features == 1536
    assert model.dino_out.in_features == 1536
    assert model.dino_out.out_features == 3072
    assert sum(parameter.numel() for parameter in model.parameters()) == 800_784_384
