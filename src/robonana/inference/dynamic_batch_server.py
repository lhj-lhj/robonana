"""Multi-client FACT TCP server with bounded-latency dynamic batching."""

from __future__ import annotations

import queue
import socket
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from world_action_model.sockets import RobotInferenceServer, _set_tcp_socket_options


@dataclass(frozen=True)
class _PendingRequest:
    observation: dict[str, Any]
    future: Future


class DynamicInferenceBatcher:
    """Collect concurrent requests and execute one policy batch at a time."""

    def __init__(self, policy, *, max_batch_size: int, max_wait_ms: float) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        if not callable(getattr(policy, "inference_batch", None)):
            raise TypeError("policy must implement inference_batch(observations)")
        self.policy = policy
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self._queue: queue.Queue[_PendingRequest | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="robonana-dynamic-batcher",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def submit(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("dynamic batcher is closed")
        future: Future = Future()
        self._queue.put(_PendingRequest(observation=observation, future=future))
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def _collect_batch(self, first: _PendingRequest) -> list[_PendingRequest]:
        batch = [first]
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                request = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if request is None:
                self._closed = True
                break
            batch.append(request)
        return batch

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                break
            batch = self._collect_batch(request)
            try:
                results = self.policy.inference_batch(
                    [item.observation for item in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"inference_batch returned {len(results)} results for "
                        f"{len(batch)} requests"
                    )
            except BaseException as error:
                for item in batch:
                    item.future.set_exception(error)
            else:
                for item, result in zip(batch, results, strict=True):
                    item.future.set_result(result)
            if self._closed:
                break


class DynamicBatchRobotInferenceServer(RobotInferenceServer):
    """FACT-compatible server that serves multiple persistent clients."""

    def __init__(
        self,
        policy,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        max_batch_size: int = 2,
        max_wait_ms: float = 6.0,
        max_clients: int = 16,
    ) -> None:
        super().__init__(policy, host=host, port=port)
        self.server_socket.listen(max(int(max_clients), int(max_batch_size)))
        self._batcher = DynamicInferenceBatcher(
            policy,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        self.register_endpoint("inference", self._batcher.submit)
        self._connection_threads: set[threading.Thread] = set()

    def run(self) -> None:
        print(
            f"Server is ready and listening on tcp://{self.host}:{self.port} "
            f"(transport=tcp, dynamic_batch={self._batcher.max_batch_size}, "
            f"wait_ms={self._batcher.max_wait_s * 1000.0:g})",
            flush=True,
        )
        try:
            while self.running:
                try:
                    conn, _ = self.server_socket.accept()
                except socket.timeout:
                    self._connection_threads = {
                        thread for thread in self._connection_threads if thread.is_alive()
                    }
                    continue
                _set_tcp_socket_options(conn)
                thread = threading.Thread(
                    target=self._serve_connection,
                    args=(conn,),
                    name="robonana-eval-client",
                    daemon=True,
                )
                self._connection_threads.add(thread)
                thread.start()
        finally:
            self.running = False
            self.server_socket.close()
            self._batcher.close()
            for thread in list(self._connection_threads):
                thread.join(timeout=5.0)
