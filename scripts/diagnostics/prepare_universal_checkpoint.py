#!/usr/bin/env python3
# 中文：离线调用 DeepSpeed 官方转换；原始 checkpoint 只读，目标必须新建。
# English: Official offline conversion; never overwrite the source checkpoint.
# 调用 / Invocation: python script --help；只写新目录 / writes a new directory only.
"""Prepare an Accelerate checkpoint for an explicitly changed DP world size.

Reference: DeepSpeed checkpoint/ds_to_universal.py (installed version owns
the conversion). No hand-written Adam partition or tensor reconstruction.
https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/checkpoint/ds_to_universal.py
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source and output must be separate non-nested directories")
    cfg = json.loads(args.source_config.read_text())
    for name in ("pytorch_model", "transformer", "scheduler.bin", "custom_checkpoint_0.pkl",
                 "target_value_expert.safetensors", "value_ema_state.json"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True, exist_ok=False)
    # Copy small metadata; hardlink immutable weights to avoid another full model.
    for p in source.iterdir():
        if p.name in {"pytorch_model", "latest", "latest_universal"}:
            continue
        if p.is_dir():
            shutil.copytree(p, output / p.name, copy_function=__import__("os").link)
        else:
            shutil.copy2(p, output / p.name)
    subprocess.run([sys.executable, "-m", "deepspeed.checkpoint.ds_to_universal",
                    "--input_folder", str(source / "pytorch_model"),
                    "--output_folder", str(output / "pytorch_model"),
                    "--num_extract_workers", "2", "--num_merge_workers", "2",
                    "--inject_missing_state"], check=True)
    # The tag is explicit in Accelerate's DeepSpeed wrapper.
    (output / "latest").write_text("pytorch_model")
    (output / "latest_universal").write_text("pytorch_model")
    ds = json.loads(Path(cfg["launch"]["deepspeed_config"]["deepspeed_config_file"]).read_text())
    ds.setdefault("checkpoint", {})["load_universal"] = True
    (output / "deepspeed_universal.json").write_text(json.dumps(ds, indent=2))
    print(json.dumps({"source": str(source), "output": str(output), "converted": True}), flush=True)


if __name__ == "__main__":
    main()
