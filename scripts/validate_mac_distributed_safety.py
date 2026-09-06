"""Two-rank integration checks for real Accelerate/ZeRO, not mocked reducers.

Run with torchrun --standalone --nproc_per_node=2. CPU checks run in pytest;
--backend deepspeed requires CUDA and is run explicitly on 190. A tiny model
isolates lifecycle semantics; real 4B training is verified separately.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin

from robonana.training.checkpointing import full_deepspeed_checkpoint
from robonana.training.posttraining import ValueExpertEMA
from robonana.training.robotwin_trainer import RoboNanaTrainer


class TinyCritics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flux = torch.nn.Linear(4, 4).requires_grad_(False)
        self.value_expert = torch.nn.Linear(4, 1)
        self.q_expert = torch.nn.Linear(4, 1)

    def forward(self, x):
        with torch.no_grad():
            hidden = self.flux(x)
        return self.value_expert(hidden) + self.q_expert(hidden)


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_same(a, e)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "deepspeed"), default="gloo")
    parser.add_argument("--mode", choices=("nonfinite", "resume"), default="nonfinite")
    parser.add_argument("--bad", choices=("nan", "inf"), default="nan")
    parser.add_argument("--micro", type=int, choices=(0, 1), default=1)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    plugin = None
    if args.backend == "deepspeed":
        plugin = DeepSpeedPlugin(hf_ds_config={
            "zero_optimization": {"stage": 2}, "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 2,
        })
    accelerator = Accelerator(cpu=args.backend == "gloo", gradient_accumulation_steps=2,
                              mixed_precision="bf16" if plugin else "no", deepspeed_plugin=plugin)
    assert accelerator.num_processes == 2
    torch.manual_seed(42)
    model = TinyCritics()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    online = accelerator.unwrap_model(model)
    trainer = object.__new__(RoboNanaTrainer)
    trainer.accelerator = accelerator
    trainer._models, trainer._optimizers, trainer._schedulers = [model], [optimizer], [scheduler]
    trainer.kwargs = {}
    trainer.with_ema = False
    trainer._cur_step = 1
    trainer.target_value_ema = ValueExpertEMA(online.value_expert, device=accelerator.device)
    trainer.current_collection_round = 0
    trainer.posttrain_config = {"phase": "critic"}
    trainer.model_name = "transformer"
    trainer.checkpoint_safe_serialization = False
    trainer.checkpoint_strict = True
    # FACT's hook uses a normal logger only on the main rank.
    import logging
    trainer.logger = logging.getLogger("safety-check")
    x = torch.ones(2, 4, device=accelerator.device, dtype=next(online.parameters()).dtype)

    def step(bad_micro=None):
        for micro in range(2):
            with accelerator.accumulate(model):
                loss = model(x).float().square().mean()
                if micro == bad_micro and accelerator.process_index == 1:
                    loss = loss * float(args.bad)
                trainer.backward_step(loss)

    step()
    assert trainer.target_value_ema.update_count == 1
    saved_model = copy.deepcopy(online.state_dict())
    saved_target = copy.deepcopy(trainer.target_value_ema.model.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    if args.mode == "nonfinite":
        caught = False
        try:
            step(args.micro)
        except FloatingPointError:
            caught = True
        caught_count = accelerator.reduce(torch.tensor(int(caught), device=accelerator.device), reduction="sum")
        assert caught_count.item() == 2
        assert_same(online.state_dict(), saved_model)
        assert_same(trainer.target_value_ema.model.state_dict(), saved_target)
        assert_same(scheduler.state_dict(), saved_scheduler)
        assert trainer.target_value_ema.update_count == 1
        # Do not resume this aborted reducer; verify all ranks terminate cleanly.
    else:
        assert args.checkpoint_dir, "resume mode requires a fresh diagnostic path"
        output = Path(args.checkpoint_dir)
        # All ranks must finish checking before any rank creates the directory.
        exists = accelerator.reduce(torch.tensor(int(output.exists()), device=accelerator.device), reduction="sum")
        assert not exists.item(), "never overwrite an existing checkpoint"
        accelerator.register_save_state_pre_hook(trainer.save_model_hook)
        accelerator.register_load_state_pre_hook(trainer.load_model_hook)
        with full_deepspeed_checkpoint(accelerator):
            accelerator.save_state(str(output), exclude_frozen_parameters=True)
        expected_rng = torch.get_rng_state().clone()
        expected_cuda_rng = torch.cuda.get_rng_state().clone() if plugin else None
        # Compare an uninterrupted next update against the resumed next update.
        # Matching weights/EMA here also exercises restored Adam/ZeRO moments
        # and FP32 master weights, not just the exported BF16 module payload.
        trainer._cur_step = 2
        step()
        uninterrupted_model = copy.deepcopy(online.state_dict())
        uninterrupted_target = copy.deepcopy(trainer.target_value_ema.model.state_dict())
        with torch.no_grad():
            for p in online.parameters():
                p.add_(10)  # Include frozen parameters: strict restore must undo this.
            for p in trainer.target_value_ema.model.parameters():
                p.zero_()
        trainer.target_value_ema.update_count = 999
        torch.rand(4)
        accelerator.load_state(str(output), **({"load_module_strict": True} if plugin else {}))
        assert_same(online.state_dict(), saved_model)
        assert_same(trainer.target_value_ema.model.state_dict(), saved_target)
        assert_same(scheduler.state_dict(), saved_scheduler)
        assert trainer.target_value_ema.update_count == 1
        assert torch.equal(torch.get_rng_state(), expected_rng)
        if plugin:
            assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)
        trainer._cur_step = 2
        step()
        assert trainer.target_value_ema.update_count == 2
        assert_same(online.state_dict(), uninterrupted_model)
        assert_same(trainer.target_value_ema.model.state_dict(), uninterrupted_target)
        assert_same(online.flux.state_dict(), {k[5:]: v for k, v in saved_model.items() if k.startswith("flux.")})
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(json.dumps(dict(status="PASS", backend=args.backend, mode=args.mode,
                              bad=args.bad, micro=args.micro)), flush=True)
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
