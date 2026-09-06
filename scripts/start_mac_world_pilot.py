"""Run the bounded world-fit pilot in a dedicated directory (usually via tmux).

Do not chain a critic job based solely on a completed world training budget:
inspect fixed-window predictions first. This driver records exact config and
source hashes, obtains the initialization baseline, trains through FACT, then
probes the final checkpoint. It never mutates replay or previous experiments.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    project = Path(os.environ["ROBONANA_PROJECT_DIR"]).resolve()
    project.mkdir(parents=True, exist_ok=False)
    from robonana.configs.robotwin_flux2_4b_mac_pilot import config
    if config["train"]["posttrain"]["phase"] != "world_policy":
        raise ValueError("start_mac_world_pilot only launches the world phase")
    manifest = dict(
        server_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        source_commit=os.environ.get("ROBONANA_SOURCE_COMMIT", "unrecorded"),
        files={str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
               for folder in ("src", "scripts") for path in sorted((repo / folder).rglob("*"))
               if path.suffix in {".py", ".sh"}},
    )
    (project / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = project / "pilot_config.json"
    snapshot.write_text(json.dumps(config, indent=2, default=str) + "\n")
    status_file = project / "pilot_status.json"
    def status(stage, **extra):
        status_file.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), **extra), indent=2) + "\n")
        print(f"pilot stage: {stage}", flush=True)
    def probe(checkpoint, model_config, name):
        command = [sys.executable, "scripts/probe_mac_world_fit.py", "--checkpoint", str(checkpoint),
                   "--model-config", str(model_config), "--data-config", str(snapshot),
                   "--output-dir", str(project / name)]
        # Training uses physical GPUs 6,7; the serial probes need just GPU 6.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["ROBONANA_GPU_IDS"].split(",")[0]}
        with (project / f"{name}.log").open("x") as log:
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status("baseline_probe")
        probe(config["models"]["checkpoint"], config["models"]["checkpoint_config"], "probe_initial")
        status("world_training")
        subprocess.run(["bash", "scripts/run_robotwin_train.sh", "--config",
                        "robonana.configs.robotwin_flux2_4b_mac_pilot.config"], cwd=repo, check=True)
        status("final_world_probe")
        checkpoints = list((project / "models").glob("checkpoint_*_step_5000/transformer/diffusion_pytorch_model.bin"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one complete step-5000 export, found {len(checkpoints)}")
        probe(checkpoints[0], project / "config.json", "probe_step5000")
        status("world_complete_review_required", checkpoint=str(checkpoints[0]))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
