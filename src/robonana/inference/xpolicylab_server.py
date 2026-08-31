"""Official XPolicyLab legacy-TCP transport for RoboNana batch evaluation."""

from __future__ import annotations

import socket
import sys
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robonana.inference.dynamic_batching import DynamicInferenceBatcher


XPOLICYLAB_CAMERA_KEYS = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
XPOLICYLAB_STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)


def xpolicylab_observation_to_robonana(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map the official XPolicyLab v1.0 observation to RoboNana's FACT layout."""

    if not isinstance(observation, Mapping):
        raise TypeError("XPolicyLab observation must be a mapping")
    vision = observation.get("vision")
    state = observation.get("state")
    if not isinstance(vision, Mapping):
        raise KeyError("XPolicyLab observation is missing the vision mapping")
    if not isinstance(state, Mapping):
        raise KeyError("XPolicyLab observation is missing the state mapping")

    converted: dict[str, Any] = {}
    for source_key, target_key in XPOLICYLAB_CAMERA_KEYS.items():
        camera = vision.get(source_key)
        if not isinstance(camera, Mapping) or "color" not in camera:
            raise KeyError(f"XPolicyLab observation is missing vision.{source_key}.color")
        image = np.asarray(camera["color"])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"vision.{source_key}.color must have shape [H,W,3], got {image.shape}"
            )
        converted[target_key] = image

    state_parts = []
    for key in XPOLICYLAB_STATE_KEYS:
        if key not in state:
            raise KeyError(f"XPolicyLab observation is missing state.{key}")
        state_parts.append(np.asarray(state[key], dtype=np.float32).reshape(-1))
    converted["observation.state"] = np.concatenate(state_parts).astype(
        np.float32, copy=False
    )

    instruction = str(
        observation.get("instruction", observation.get("prompt", ""))
    ).strip()
    if not instruction:
        instructions = observation.get("instructions")
        if isinstance(instructions, Sequence) and not isinstance(
            instructions, (str, bytes, bytearray)
        ) and instructions:
            instruction = str(instructions[0]).strip()
    if not instruction:
        raise ValueError("XPolicyLab observation instruction is empty")
    converted["instruction"] = instruction
    return converted


def load_xpolicylab_transport(
    xpolicylab_root: str | Path,
) -> tuple[Callable[[Any], str], Callable[[str], Any], Callable[[Any], Any]]:
    """Load codecs directly from the checked-out official XPolicyLab tree."""

    root = Path(xpolicylab_root).expanduser().resolve()
    if not (root / "client_server" / "tcp" / "utils.py").is_file():
        raise FileNotFoundError(f"invalid XPolicyLab root: {root}")
    for import_root in (root.parent, root):
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.insert(0, import_root_text)
    from client_server.tcp.utils import json_to_numpy, numpy_to_json
    from XPolicyLab.utils.process_data import decode_obs_images

    return numpy_to_json, json_to_numpy, decode_obs_images


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = connection.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ConnectionError("connection closed while receiving a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class XPolicyLabDynamicBatchServer:
    """Serve concurrent official XPolicyLab clients with one RoboNana batcher.

    Each official batch-eval worker owns one persistent TCP connection. Its
    latest observation therefore stays connection-local, while simultaneous
    ``get_action`` calls are coalesced into one ``policy.inference_batch``.
    """

    def __init__(
        self,
        policy,
        *,
        xpolicylab_root: str | Path,
        host: str = "127.0.0.1",
        port: int = 8094,
        max_batch_size: int = 8,
        max_wait_ms: float = 100.0,
        max_clients: int = 32,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.max_clients = int(max_clients)
        self._encode, self._decode, self._decode_obs_images = load_xpolicylab_transport(
            xpolicylab_root
        )
        self._batcher = DynamicInferenceBatcher(
            policy,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(max(self.max_clients, max_batch_size))
        self._socket.settimeout(1.0)
        self.port = int(self._socket.getsockname()[1])
        self._running = False
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()

    def run(self) -> None:
        self._running = True
        print(
            f"XPolicyLab legacy-TCP server listening on {self.host}:{self.port} "
            f"(dynamic_batch={self._batcher.max_batch_size}, "
            f"wait_ms={self._batcher.max_wait_s * 1000.0:g})",
            flush=True,
        )
        try:
            while self._running:
                try:
                    connection, _ = self._socket.accept()
                except socket.timeout:
                    self._discard_finished_threads()
                    continue
                except OSError:
                    if self._running:
                        raise
                    break
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                thread = threading.Thread(
                    target=self._serve_connection,
                    args=(connection,),
                    name="robonana-xpolicylab-client",
                    daemon=True,
                )
                with self._threads_lock:
                    self._threads.add(thread)
                thread.start()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._running:
            self._running = False
            try:
                self._socket.close()
            except OSError:
                pass
        self._batcher.close()
        with self._threads_lock:
            threads = list(self._threads)
        current = threading.current_thread()
        for thread in threads:
            if thread is not current:
                thread.join(timeout=5.0)

    def _discard_finished_threads(self) -> None:
        with self._threads_lock:
            self._threads = {thread for thread in self._threads if thread.is_alive()}

    @staticmethod
    def _extract_action(result: Any) -> Any:
        if isinstance(result, Mapping) and "action" in result:
            return result["action"]
        return result

    def _serve_connection(self, connection: socket.socket) -> None:
        latest_observation: dict[str, Any] | None = None
        batch_observations: dict[int, dict[str, Any]] = {}
        try:
            with connection:
                while self._running:
                    header = connection.recv(4)
                    if not header:
                        break
                    if len(header) != 4:
                        header += _recv_exact(connection, 4 - len(header))
                    request = self._decode(
                        _recv_exact(connection, int.from_bytes(header, "big")).decode(
                            "utf-8"
                        )
                    )
                    command = request.get("cmd")
                    payload = request.get("obs")
                    if payload is not None:
                        payload = self._decode_obs_images(payload)

                    if command == "reset":
                        latest_observation = None
                        batch_observations.clear()
                        result = None
                    elif command == "update_obs":
                        latest_observation = xpolicylab_observation_to_robonana(payload)
                        result = None
                    elif command == "get_action":
                        if latest_observation is None:
                            raise RuntimeError("get_action called before update_obs")
                        result = self._extract_action(
                            self._batcher.submit(latest_observation)
                        )
                    elif command == "update_obs_batch":
                        if not isinstance(payload, Sequence):
                            raise TypeError("update_obs_batch expects an observation sequence")
                        for index, observation in enumerate(payload):
                            env_index = int(observation.get("env_idx", index))
                            batch_observations[env_index] = (
                                xpolicylab_observation_to_robonana(observation)
                            )
                        result = None
                    elif command == "get_action_batch":
                        env_indices = [int(index) for index in payload]
                        missing = [
                            index for index in env_indices if index not in batch_observations
                        ]
                        if missing:
                            raise RuntimeError(
                                f"get_action_batch has no observation for env indices {missing}"
                            )
                        results = self._batcher.submit_many(
                            [batch_observations[index] for index in env_indices]
                        )
                        result = [self._extract_action(item) for item in results]
                    elif command in {"prepare_case", "trial_end"}:
                        result = None
                    else:
                        raise AttributeError(f"unsupported XPolicyLab command: {command!r}")
                    self._send(connection, {"res": result})
        except (ConnectionError, ConnectionResetError, BrokenPipeError, OSError):
            return
        except Exception as error:
            try:
                self._send(
                    connection,
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                )
            except OSError:
                pass

    def _send(self, connection: socket.socket, payload: Any) -> None:
        encoded = self._encode(payload).encode("utf-8")
        connection.sendall(len(encoded).to_bytes(4, "big"))
        connection.sendall(encoded)
