"""Multi-client FACT TCP server with bounded-latency dynamic batching."""

from __future__ import annotations

import socket
import threading

from robonana.inference.dynamic_batching import DynamicInferenceBatcher
from world_action_model.sockets import RobotInferenceServer, _set_tcp_socket_options


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
