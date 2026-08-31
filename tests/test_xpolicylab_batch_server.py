from concurrent.futures import ThreadPoolExecutor
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robonana.inference.xpolicylab_server import (
    XPolicyLabDynamicBatchServer,
    xpolicylab_observation_to_robonana,
)


def _xpl_observation(env_idx: int = 0) -> dict:
    return {
        "data_format_version": "v1.0",
        "instruction": "pick up the cube",
        "env_idx": env_idx,
        "vision": {
            "cam_head": {"color": np.full((8, 12, 3), env_idx, dtype=np.uint8)},
            "cam_left_wrist": {"color": np.full((4, 6, 3), env_idx + 1, dtype=np.uint8)},
            "cam_right_wrist": {"color": np.full((4, 6, 3), env_idx + 2, dtype=np.uint8)},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.asarray([6], dtype=np.float32),
            "right_arm_joint_state": np.arange(7, 13, dtype=np.float32),
            "right_ee_joint_state": np.asarray([13], dtype=np.float32),
        },
    }


def test_xpolicylab_observation_conversion_matches_training_layout() -> None:
    converted = xpolicylab_observation_to_robonana(_xpl_observation(3))
    assert converted["instruction"] == "pick up the cube"
    np.testing.assert_array_equal(
        converted["observation.state"], np.arange(14, dtype=np.float32)
    )
    assert converted["observation.images.cam_high"].shape == (8, 12, 3)
    assert converted["observation.images.cam_left_wrist"][0, 0, 0] == 4
    assert converted["observation.images.cam_right_wrist"][0, 0, 0] == 5


def test_official_legacy_clients_coalesce_across_connections(tmp_path: Path) -> None:
    xpl_root = tmp_path / "XPolicyLab"
    codec_dir = xpl_root / "client_server" / "tcp"
    process_dir = xpl_root / "XPolicyLab" / "utils"
    codec_dir.mkdir(parents=True)
    process_dir.mkdir(parents=True)
    for package in (
        xpl_root / "client_server",
        codec_dir,
        xpl_root / "XPolicyLab",
        process_dir,
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (codec_dir / "utils.py").write_text(
        "import json, numpy as np\n"
        "class E(json.JSONEncoder):\n"
        " def default(self, x):\n"
        "  if isinstance(x, np.ndarray): return {'__a__': x.tolist()}\n"
        "  try:\n"
        "   import torch\n"
        "   if isinstance(x, torch.Tensor): return {'__a__': x.detach().cpu().tolist()}\n"
        "  except ImportError: pass\n"
        "  return super().default(x)\n"
        "def numpy_to_json(x): return json.dumps(x, cls=E)\n"
        "def json_to_numpy(x): return json.loads(x, object_hook=lambda d: np.asarray(d['__a__']) if '__a__' in d else d)\n",
        encoding="utf-8",
    )
    (process_dir / "process_data.py").write_text(
        "def decode_obs_images(x): return x\n", encoding="utf-8"
    )

    class Policy:
        def __init__(self) -> None:
            self.batch_sizes = []

        def inference_batch(self, observations):
            self.batch_sizes.append(len(observations))
            time.sleep(0.01)
            return [
                {
                    "action": np.full(
                        (2, 14), obs["observation.images.cam_high"][0, 0, 0]
                    )
                }
                for obs in observations
            ]

    policy = Policy()
    server = XPolicyLabDynamicBatchServer(
        policy,
        xpolicylab_root=xpl_root,
        host="127.0.0.1",
        port=0,
        max_batch_size=4,
        max_wait_ms=50,
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    sys.path.insert(0, str(xpl_root))
    from client_server.tcp.utils import json_to_numpy, numpy_to_json

    barrier = threading.Barrier(4)

    def call(command, payload, connection):
        request = numpy_to_json({"cmd": command, "obs": payload}).encode()
        connection.sendall(len(request).to_bytes(4, "big") + request)
        size = int.from_bytes(connection.recv(4), "big")
        response = bytearray()
        while len(response) < size:
            response.extend(connection.recv(size - len(response)))
        return json_to_numpy(response.decode())["res"]

    def client(index: int):
        with socket.create_connection(("127.0.0.1", server.port), timeout=2) as connection:
            call("reset", None, connection)
            call("update_obs", _xpl_observation(index), connection)
            barrier.wait(timeout=2)
            return call("get_action", None, connection)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(client, range(4)))
    finally:
        server.stop()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert policy.batch_sizes == [4]
    for index, result in enumerate(results):
        assert result.shape == (2, 14)
        assert np.all(result == index)
