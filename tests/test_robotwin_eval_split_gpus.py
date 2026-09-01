from pathlib import Path


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


def test_hanging_mug_fresh_round_combines_eval_and_replay_collection() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_hanging_mug_posttrain_round.sh"
    ).read_text(encoding="utf-8")

    combined_guard = (
        "if [[ ! -f ${state_dir}/pretrain_eval.done \\\n"
        "  && ! -f ${state_dir}/rollout_replay.done ]]; then"
    )
    assert combined_guard in script
    assert 'collect_and_prepare_replay 1 "${pre_eval_dir}"' in script
    assert script.index('touch "${state_dir}/pretrain_eval.done"') < script.index(
        'touch "${state_dir}/rollout_replay.done"'
    )
