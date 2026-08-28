from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from robonana.inference.dynamic_batch_server import DynamicBatchRobotInferenceServer
from world_action_model.sockets import RobotInferenceClient


def test_many_persistent_clients_do_not_drop_or_cross_wire_requests() -> None:
    client_count = 16
    requests_per_client = 10
    max_batch_size = 8

    class StressPolicy:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.lock = threading.Lock()

        def inference_batch(self, observations):
            with self.lock:
                self.batch_sizes.append(len(observations))
            time.sleep(0.005)
            return [
                {"request_id": observation["request_id"]}
                for observation in observations
            ]

    policy = StressPolicy()
    server = DynamicBatchRobotInferenceServer(
        policy,
        host="127.0.0.1",
        port=0,
        max_batch_size=max_batch_size,
        max_wait_ms=20,
        max_clients=client_count + 2,
    )
    port = server.server_socket.getsockname()[1]
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    barrier = threading.Barrier(client_count)

    def run_client(client_index: int) -> list[int]:
        client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=5000)
        try:
            barrier.wait(timeout=5)
            outputs = []
            for request_index in range(requests_per_client):
                request_id = client_index * 10_000 + request_index
                response = client.inference({"request_id": request_id})
                outputs.append(int(response["request_id"]))
            return outputs
        finally:
            client.close()

    try:
        with ThreadPoolExecutor(max_workers=client_count) as executor:
            futures = [executor.submit(run_client, index) for index in range(client_count)]
            outputs = [value for future in futures for value in future.result(timeout=20)]
    finally:
        stop_client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=2000)
        try:
            stop_client.kill_server()
        finally:
            stop_client.close()
        server_thread.join(timeout=5)

    expected = [
        client_index * 10_000 + request_index
        for client_index in range(client_count)
        for request_index in range(requests_per_client)
    ]
    assert not server_thread.is_alive()
    assert Counter(outputs) == Counter(expected)
    assert sum(policy.batch_sizes) == client_count * requests_per_client
    assert all(1 <= size <= max_batch_size for size in policy.batch_sizes)
    assert max_batch_size in policy.batch_sizes


def test_incomplete_tail_batch_flushes_after_wait_window() -> None:
    class TailPolicy:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def inference_batch(self, observations):
            self.batch_sizes.append(len(observations))
            return [{"request_id": item["request_id"]} for item in observations]

    policy = TailPolicy()
    server = DynamicBatchRobotInferenceServer(
        policy,
        host="127.0.0.1",
        port=0,
        max_batch_size=8,
        max_wait_ms=25,
        max_clients=5,
    )
    port = server.server_socket.getsockname()[1]
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    barrier = threading.Barrier(3)

    def infer(request_id: int):
        client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=2000)
        try:
            barrier.wait(timeout=2)
            return client.inference({"request_id": request_id})
        finally:
            client.close()

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(infer, range(3)))
    finally:
        stop_client = RobotInferenceClient(host="127.0.0.1", port=port, timeout_ms=2000)
        try:
            stop_client.kill_server()
        finally:
            stop_client.close()
        server_thread.join(timeout=5)
    elapsed = time.perf_counter() - started

    assert [result["request_id"] for result in results] == [0, 1, 2]
    assert policy.batch_sizes == [3]
    assert elapsed < 1.0
