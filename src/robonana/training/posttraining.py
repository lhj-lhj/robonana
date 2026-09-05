"""Value-only target updates for the maintained MAC critic phase."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn


@contextmanager
def evaluating(model: nn.Module) -> Iterator[None]:
    """Temporarily switch a model to eval without changing its gradients."""

    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class ValueExpertEMA:
    """FP32 target copy of Value only; FLUX and Q are never duplicated.

    This target topology follows MAC: the target/EMA network exists only for
    Value, while Q bootstraps from the online Value and action selection uses
    the online Q function.  See the original implementation:
    https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L191-L217
    https://github.com/kwanyoungpark/MAC/blob/main/agents/mac.py#L262-L318
    """

    def __init__(
        self,
        online_expert: nn.Module,
        *,
        decay: float = 0.995,
        update_every_optimizer_steps: int = 1,
        start_step: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        if int(update_every_optimizer_steps) <= 0:
            raise ValueError("EMA update interval must be positive")
        if int(start_step) < 0:
            raise ValueError("EMA start_step cannot be negative")
        self.decay = float(decay)
        self.update_every_optimizer_steps = int(update_every_optimizer_steps)
        self.start_step = int(start_step)
        self.update_count = 0
        self.last_online_l2 = 0.0
        self.model = copy.deepcopy(online_expert)
        if device is not None:
            self.model.to(device)
        self.model.float().eval().requires_grad_(False)

    @torch.no_grad()
    def exact_copy_from(self, online_expert: nn.Module) -> None:
        online = online_expert.state_dict()
        target = self.model.state_dict()
        if online.keys() != target.keys():
            raise RuntimeError("online and target Value expert state keys differ")
        for name, destination in target.items():
            destination.copy_(online[name].detach().to(destination))
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        online_expert: nn.Module,
        *,
        optimizer_step: int,
        optimizer_step_succeeded: bool,
    ) -> bool:
        if not optimizer_step_succeeded:
            return False
        optimizer_step = int(optimizer_step)
        if optimizer_step < self.start_step:
            return False
        if (optimizer_step - self.start_step) % self.update_every_optimizer_steps:
            return False
        online_parameters = dict(online_expert.named_parameters())
        squared = torch.zeros((), device=next(self.model.parameters()).device, dtype=torch.float64)
        count = 0
        for name, target in self.model.named_parameters():
            source = online_parameters[name].detach().to(device=target.device, dtype=torch.float32)
            target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            squared.add_((target.double() - source.double()).square().sum())
            count += source.numel()
        online_buffers = dict(online_expert.named_buffers())
        for name, target in self.model.named_buffers():
            target.copy_(online_buffers[name].detach().to(target))
        self.update_count += 1
        self.last_online_l2 = float(torch.sqrt(squared / max(count, 1)).cpu())
        self.model.eval().requires_grad_(False)
        return True

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().float().contiguous()
            for name, value in self.model.state_dict().items()
            if isinstance(value, Tensor)
        }

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        destination = self.model.state_dict()
        if destination.keys() != state_dict.keys():
            missing = sorted(destination.keys() - state_dict.keys())
            unexpected = sorted(state_dict.keys() - destination.keys())
            raise RuntimeError(
                f"target Value checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.model.load_state_dict(
            {
                name: value.to(device=destination[name].device, dtype=destination[name].dtype)
                for name, value in state_dict.items()
            },
            strict=True,
        )
        self.model.eval().requires_grad_(False)

    def assert_not_in_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        overlap = [
            name for name, parameter in self.model.named_parameters()
            if id(parameter) in optimizer_ids
        ]
        if overlap:
            raise RuntimeError(f"target Value parameters leaked into optimizer: {overlap[:5]}")
