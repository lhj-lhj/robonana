from __future__ import annotations


def test_dotted_robotwin_policy_adapter_exports_fact_hooks() -> None:
    from robonana_robotwin import adapter

    assert callable(adapter.get_model)
    assert callable(adapter.eval)
    assert callable(adapter.reset_model)


def test_small200m_robotwin_variant_matches_training_config() -> None:
    from robonana.inference import robotwin_model_params

    params = robotwin_model_params("small200m")

    assert params.hidden_size == 1024
    assert params.num_heads == 8
    assert params.depth == 2
    assert params.depth_single_blocks == 8
    assert params.context_in_dim == 7680
