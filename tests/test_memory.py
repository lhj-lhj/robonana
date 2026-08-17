import pytest

from robonana.training.memory import GIB, estimate_required_free_bytes, memory_preflight


def test_adapter_and_full_preflight_have_distinct_memory_bounds():
    checkpoint = 8 * GIB
    assert estimate_required_free_bytes(checkpoint, "adapters") == 10 * GIB
    assert estimate_required_free_bytes(checkpoint, "full") == 34 * GIB
    assert memory_preflight(checkpoint_bytes=checkpoint, free_bytes=14 * GIB, mode="adapters").can_run
    assert not memory_preflight(checkpoint_bytes=checkpoint, free_bytes=14 * GIB, mode="full").can_run


def test_preflight_rejects_unknown_mode():
    with pytest.raises(ValueError, match="train mode"):
        estimate_required_free_bytes(GIB, "unknown")
