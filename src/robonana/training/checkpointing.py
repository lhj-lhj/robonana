"""Small FACT/Accelerate checkpoint-policy adapter; no trainer fork."""

from contextlib import contextmanager

from accelerate.utils import DistributedType


@contextmanager
def full_deepspeed_checkpoint(accelerator):
    """Keep frozen parameters in ZeRO's module payload for strict resume.

    FACT currently hard-codes ``exclude_frozen_parameters=True`` in its
    save_checkpoint_step. A separate full transformer export does not repair
    the ZeRO payload: Accelerate restores the DeepSpeed engine before invoking
    our load hook. Override only this argument while FACT owns the save, and
    retain its optimizer/scheduler/RNG/hook lifecycle unchanged.

    https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#deepspeed.DeepSpeedEngine.save_checkpoint
    """
    if getattr(accelerator, "distributed_type", None) != DistributedType.DEEPSPEED:
        yield
        return
    save_state = accelerator.save_state

    def save_full_state(*args, **kwargs):
        kwargs["exclude_frozen_parameters"] = False
        return save_state(*args, **kwargs)

    accelerator.save_state = save_full_state
    try:
        yield
    finally:
        # Also restore the entry point on disk/collective failures. The adapter
        # is scoped to the synchronous trainer save, never a permanent patch.
        accelerator.save_state = save_state
