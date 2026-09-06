from types import SimpleNamespace

import pytest
from accelerate.utils import DistributedType

from robonana.training.checkpointing import full_deepspeed_checkpoint


@pytest.mark.parametrize("fail", [False, True])
def test_full_zero_save_overrides_fact_argument_and_restores_method(fail):
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        if fail:
            raise OSError("disk failure")
    accelerator = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED, save_state=original)
    try:
        with full_deepspeed_checkpoint(accelerator):
            accelerator.save_state("checkpoint", exclude_frozen_parameters=True, tag="step2")
    except OSError:
        assert fail
    assert calls == [(("checkpoint",), {"exclude_frozen_parameters": False, "tag": "step2"})]
    assert accelerator.save_state is original


def test_non_zero_backend_is_not_patched():
    original = lambda *args: None
    accelerator = SimpleNamespace(distributed_type=DistributedType.MULTI_CPU, save_state=original)
    with full_deepspeed_checkpoint(accelerator):
        assert accelerator.save_state is original
