from pathlib import Path

import json
import os
import shutil
import subprocess
import sys

import pytest


def test_eval_launcher_requires_disjoint_policy_and_simulator_gpu_pools() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_robotwin_all_tasks_parallel.sh"
    ).read_text(encoding="utf-8")

    assert "ROBONANA_EVAL_SERVER_GPUS" in script
    assert "ROBONANA_EVAL_SIM_GPUS" in script
    assert "the pools must be disjoint" in script
    assert 'CUDA_VISIBLE_DEVICES="${server_gpu}"' in script
    assert '"CUDA_VISIBLE_DEVICES=${sim_gpu}"' in script
    assert '"OIDN_DEFAULT_DEVICE=cuda"' in script


def test_rollout_collector_separates_policy_and_simulator_gpus() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "collect_prepare_robotwin_rollouts.sh"
    ).read_text(encoding="utf-8")

    assert "ROBONANA_SERVER_GPU_ID" in script
    assert "ROBONANA_SIM_GPU_ID" in script
    assert "policy server and SAPIEN simulator GPUs must be disjoint" in script
    assert 'ROBONANA_EVAL_SERVER_GPUS="${server_gpu_id}"' in script
    assert 'ROBONANA_EVAL_SIM_GPUS="${sim_gpu_id}"' in script
    assert 'bash "${isolated_eval}"' in script
    assert 'eval_run_dir=${ROBONANA_EVAL_RUN_DIR:-' in script
    assert 'video_log=${EVAL_VIDEO_LOG:-0}' in script
    assert 'ROBONANA_EVAL_RUN_DIR="${eval_run_dir}"' in script
    assert 'EVAL_VIDEO_LOG="${video_log}"' in script
    assert "ROBONANA_ROBOTWIN_STATIC_CAMERAS" in script


def test_hanging_mug_round_serializes_world_then_critic_then_collection() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_hanging_mug_mac_round.sh"
    ).read_text(encoding="utf-8")

    assert "ROBONANA_MAC_PHASE=world_policy" in script
    assert "ROBONANA_MAC_PHASE=critic" in script
    assert "ROBONANA_MAC_TARGET_VALUE_CHECKPOINT" not in script
    assert "ROBONANA_MAC_TARGET_VALUE_STATE" not in script
    assert script.index('touch "${state_dir}/world_policy.done"') < script.index(
        'touch "${state_dir}/critic.done"'
    )
    assert script.index('touch "${state_dir}/critic.done"') < script.index(
        'run_action_only_eval "${m1_eval_dir}"'
    )
    assert "ROBONANA_INFERENCE_MODE=action_only" in script
    assert 'ROBONANA_REJECTION_CANDIDATE_COUNT="${candidate_count}"' not in script
    # Collection is a separate 100-episode budget, not the comparison eval size.
    assert 'collection_num=${ROBONANA_MAC_COLLECTION_EPISODES:-100}' in script
    assert 'TEST_NUM="${collection_num}"' in script
    collector = (Path(__file__).resolve().parents[1] / "scripts/collect_prepare_robotwin_rollouts.sh").read_text(encoding="utf-8")
    assert 'test_num=${TEST_NUM:-${ROBONANA_MAC_COLLECTION_EPISODES:-100}}' in collector


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Linux launcher integration")
@pytest.mark.parametrize("failure", ["", "world_exit", "world_truncated", "critic_truncated"])
def test_training_only_chains_complete_phases_without_simulator(tmp_path, failure):
    """Run the real shell orchestrator with a cheap checkpoint-writing trainer.

    中文：覆盖自动串联、前一阶段失败/磁盘截断时不进入下一阶段，且无仿真依赖。
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/run_hanging_mug_mac_round.sh"
    shutil.copyfile(source, scripts / source.name)
    fake = scripts / "trainer.py"
    fake.write_text('''import json, os, sys, zipfile
from pathlib import Path
phase = os.environ["ROBONANA_MAC_PHASE"]
root = Path(os.environ["ROBONANA_PROJECT_DIR"])
root.mkdir(parents=True, exist_ok=True)
with (root.parent / "calls.jsonl").open("a") as f:
    f.write(json.dumps(dict(phase=phase, source=os.environ["ROBONANA_MAC_PRETRAIN_CHECKPOINT"],
                           resume=os.environ["ROBONANA_RESUME"], steps=os.environ["ROBONANA_MAX_STEPS"])) + "\\n")
if phase == "world_policy" and os.environ["TEST_FAILURE"] == "world_exit":
    sys.exit(3)
ck = root / ("models/checkpoint_epoch_1_step_" + os.environ["ROBONANA_MAX_STEPS"])
(ck / "transformer").mkdir(parents=True)
(ck / "pytorch_model").mkdir()
(root / "config.json").write_text(json.dumps(dict(launch=dict(gpu_ids=[0,1]), train=dict(posttrain=dict(phase=phase)))))
for name in ("transformer/diffusion_pytorch_model.bin", "pytorch_model/mp_rank_00_model_states.pt",
             "pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt", "pytorch_model/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"):
    with zipfile.ZipFile(ck / name, "w") as f:
        f.writestr("tensor", "fixture")
for name in ("scheduler.bin", "custom_checkpoint_0.pkl", "random_states_0.pkl", "random_states_1.pkl",
             "target_value_expert.safetensors", "value_ema_state.json"):
    (ck / name).write_text("fixture")
(ck / "transformer/inference_contract.json").write_text(json.dumps(dict(step=int(os.environ["ROBONANA_MAX_STEPS"]))))
if os.environ["TEST_FAILURE"] == ("world_truncated" if phase == "world_policy" else "critic_truncated"):
    (ck / "transformer/diffusion_pytorch_model.bin").write_bytes(b"truncated")
''', encoding="utf-8")
    (scripts / "run_robotwin_train.sh").write_text(
        '#!/usr/bin/env bash\nset -e\n"$ROBONANA_MODEL_PYTHON" "$(dirname "$0")/trainer.py"\n', encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "robonana_norm_stats.json").write_text("{}")
    initial = tmp_path / "step3000.bin"
    initial.write_text("fixture")
    config = tmp_path / "initial.json"
    config.write_text("{}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROBONANA_")}
    env.update(ROBONANA_MAC_TRAIN_ONLY="1", ROBONANA_RESUME="0", TEST_FAILURE=failure,
               ROBONANA_MAC_WORLD_POLICY_STEPS="10000", ROBONANA_MAC_CRITIC_STEPS="10000",
               ROBONANA_MODEL_PYTHON=sys.executable, ROBONANA_INITIAL_DATASET_ROOT=str(data),
               ROBONANA_MAC_SOURCE_CHECKPOINT=str(initial), ROBONANA_MAC_SOURCE_CONFIG=str(config),
               ROBONANA_REPLAY_ROOT=str(data), ROBONANA_PROJECT_DIR=str(tmp_path / "run"),
               ROBONANA_MAC_RUN_ROOT=str(tmp_path / "out"), ROBONANA_ROBOTWIN_PYTHON="/missing/simulator",
               ROBONANA_MIN_TRAIN_FREE_GIB="0")
    result = subprocess.run(["bash", str(scripts / source.name)], env=env, capture_output=True, text=True, timeout=30)
    calls = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert (result.returncode == 0) == (not failure), result.stdout + result.stderr
    assert calls[0] == dict(phase="world_policy", source=str(initial), resume="0", steps="10000")
    assert len(calls) == (1 if failure.startswith("world_") else 2)
    if len(calls) == 2:
        assert calls[1] == dict(phase="critic", source=str(tmp_path / "run/world_policy/models/checkpoint_epoch_1_step_10000/transformer/diffusion_pytorch_model.bin"), resume="0", steps="10000")
    assert not (tmp_path / "out/state/critic.done").exists() if failure else (tmp_path / "out/state/critic.done").exists()
    assert not (tmp_path / "out/m1_eval").exists()
