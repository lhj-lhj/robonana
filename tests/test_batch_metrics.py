import json
import pytest

from robonana.inference.dynamic_batching import BatchMetricsPolicy


def test_metrics_preserve_results_order_and_delegate_attributes(tmp_path):
    class Policy:
        marker = object()
        def inference_batch(self, observations):
            return [{"action": item["sampling_seed"]} for item in observations]
    path = tmp_path / "metrics.jsonl"
    policy = Policy()
    wrapped = BatchMetricsPolicy(policy, path)
    assert wrapped.marker is policy.marker
    assert wrapped.inference_batch([{"sampling_seed": 2}, {"sampling_seed": 1}]) == [
        {"action": 2}, {"action": 1}]
    wrapped.inference_batch([{"sampling_seed": 3}])
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    assert [row["batch_size"] for row in rows] == [2, 1]
    assert rows[0]["sampling_seeds"] == [2, 1]
    assert all(row["wall_ms"] >= 0 for row in rows)
    with pytest.raises(FileExistsError):
        BatchMetricsPolicy(policy, path)
