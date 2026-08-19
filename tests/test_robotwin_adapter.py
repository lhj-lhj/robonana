from __future__ import annotations


def test_dotted_robotwin_policy_adapter_exports_fact_hooks() -> None:
    from robonana_robotwin import adapter

    assert callable(adapter.get_model)
    assert callable(adapter.eval)
    assert callable(adapter.reset_model)
