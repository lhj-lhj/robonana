"""Checkpoint-bound inference settings; no inference defaults or cache retro-certification.

The sidecar lives beside the exported transformer weights and hashes those
exact bytes. A run-level config or a neighbouring checkpoint is not evidence.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

CONTRACT_FILE = "inference_contract.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_contract(posttrain):
    """Train and environment candidate budgets are intentionally distinct."""
    imagination = posttrain["imagination"]
    environment = posttrain["environment_policy"]
    if type(imagination["candidate_count"]) is not int or imagination["candidate_count"] <= 0:
        raise ValueError("Imagination candidate_count must be a positive integer")
    result = dict(
        num_inference_steps=imagination["sampling_steps"], flow_shift=imagination["flow_shift"],
        rejection_candidate_count=environment["candidate_count"],
        action_chunk=environment["action_chunk"], horizon=posttrain["chunk_horizon"],
        discount=posttrain["discount"], reward_non_goal=posttrain["reward_non_goal"],
        reward_goal=posttrain["reward_goal"], q_return_scale=posttrain["return_scale"],
        success_threshold=0.5,
    )
    for key in ("num_inference_steps", "rejection_candidate_count"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("Inference settings must be finite")
    if result["flow_shift"] <= 0 or result["q_return_scale"] <= 0 or not 0 < result["discount"] <= 1:
        raise ValueError("Invalid flow_shift, return_scale or discount")
    if result["action_chunk"] != 48 or result["horizon"] != 48 or environment["execute_actions_per_plan"] != 48:
        raise ValueError("Inference contract requires fixed/executed horizon 48")
    if imagination["candidate_selection"] != "argmax_q" or environment["candidate_selection"] != "argmax_q":
        raise ValueError("Inference contract requires argmax_q")
    return result


def build_contract(posttrain, vae_checkpoint):
    from robonana.image_pipeline import image_contract
    from robonana.normalization import require_a_stats_path
    return dict(
        version=1, sampling=sampling_contract(posttrain),
        imagination_candidate_count=posttrain["imagination"]["candidate_count"],
        image=image_contract(str(Path(vae_checkpoint).expanduser().resolve())),
        normalization_sha256=sha256_file(require_a_stats_path()),
        action_mapping="A_zscore_delta_to_absolute_no_clip_nonfinite_fallback_v1",
    )


def write_contract(weights, contract, *, phase, step):
    payload = dict(contract, weights_sha256=sha256_file(weights), phase=phase, step=step)
    path = Path(weights).parent / CONTRACT_FILE
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_contract(weights):
    path = Path(weights).parent / CONTRACT_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Uncertified checkpoint: missing {path}. Cache contracts/config.json cannot certify old weights."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "sampling", "image", "normalization_sha256", "action_mapping",
                "imagination_candidate_count", "weights_sha256", "phase", "step"}
    if not required <= payload.keys() or payload["version"] != 1:
        raise ValueError(f"Incomplete/unsupported inference contract: {path}")
    if payload["weights_sha256"] != sha256_file(weights):
        raise ValueError(f"Checkpoint weight fingerprint mismatch: {path}")
    return payload


def check_contract(actual, expected):
    if actual.get('phase') == 'converted_action_only':
        raise ValueError('Converted actor is not a certified training/critic checkpoint; explicitly adapt in Stage 1')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Checkpoint inference contract mismatch: {key}")


def resolve_online_contract(weights, vae_checkpoint, overrides, *, inference_mode=None):
    from robonana.image_pipeline import image_contract
    from robonana.normalization import require_a_stats_path
    actual = read_contract(weights)
    execution = actual
    if actual.get('phase') == 'converted_action_only':
        if inference_mode != 'action_only' or actual.get('capabilities') != ['action_only']:
            raise ValueError('Converted 120k actor only supports action_only; Q/world heads are untrained')
        # Explicit execution-only migration, never historical input certification.
        execution = dict(actual, phase='execution_validation')
    check_contract(execution, dict(
        image=image_contract(str(Path(vae_checkpoint).expanduser().resolve())),
        normalization_sha256=sha256_file(require_a_stats_path()),
        action_mapping="A_zscore_delta_to_absolute_no_clip_nonfinite_fallback_v1",
    ))
    settings = copy.deepcopy(actual["sampling"])
    if set(overrides) != set(settings):
        raise ValueError("Incomplete checkpoint sampling schema")
    for key, requested in overrides.items():
        if requested is not None and requested != settings[key]:
            raise ValueError(f"Checkpoint sampling mismatch: {key}: requested={requested}, saved={settings[key]}")
    return actual
