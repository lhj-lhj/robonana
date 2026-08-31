"""Dependency-free bounded-latency batching shared by inference transports."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


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

    def submit_many(
        self, observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Queue a caller-owned batch without serially waiting on each item."""

        if self._closed:
            raise RuntimeError("dynamic batcher is closed")
        if not observations:
            return []
        futures = [Future() for _ in observations]
        for observation, future in zip(observations, futures, strict=True):
            self._queue.put(_PendingRequest(observation=observation, future=future))
        return [future.result() for future in futures]

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
