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
