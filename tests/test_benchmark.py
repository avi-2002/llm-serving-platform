from typing import Any

import pytest

from low_latency_llm_serving.benchmark import (
    BenchmarkConfig,
    RequestRecord,
    execute_request,
    percentile,
    summarize_level,
)


def make_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        base_url="http://127.0.0.1:8000",
        prompt="Explain latency.",
        max_new_tokens=16,
        requests_per_level=3,
        concurrency_levels=(1, 2),
        warmup_requests=0,
        timeout_seconds=10,
    )


def test_percentile_interpolates_ordered_values() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 50) == 2.5
    assert percentile(values, 95) == pytest.approx(3.85)


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 50)


def test_summary_calculates_throughput_percentiles_and_errors() -> None:
    records = [
        RequestRecord("a", 2, True, 200, 1.0, 0.8, 0.9, 10, 8, None),
        RequestRecord("b", 2, True, 200, 2.0, 0.9, 1.9, 10, 8, None),
        RequestRecord("c", 2, False, 503, 0.1, None, None, None, None, "busy"),
    ]

    summary = summarize_level(records, concurrency=2, wall_seconds=2.0)

    assert summary.successful_requests == 2
    assert summary.failed_requests == 1
    assert summary.error_rate == pytest.approx(1 / 3)
    assert summary.requests_per_second == 1.0
    assert summary.output_tokens_per_second == 8.0
    assert summary.client_latency_p50_seconds == 1.5
    assert summary.server_overhead_mean_seconds == pytest.approx(0.55)
    assert summary.replicas_observed == 0


def test_summary_counts_distinct_ray_replicas() -> None:
    records = [
        RequestRecord(
            "a", 2, True, 200, 1.0, 0.8, 0.9, 10, 8, None, "replica-a"
        ),
        RequestRecord(
            "b", 2, True, 200, 1.0, 0.8, 0.9, 10, 8, None, "replica-b"
        ),
    ]

    summary = summarize_level(records, concurrency=2, wall_seconds=1.0)

    assert summary.replicas_observed == 2


def test_execute_request_maps_successful_response() -> None:
    def fake_post(
        url: str, payload: dict[str, object], request_id: str, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        assert url.endswith("/v1/generate")
        assert payload["do_sample"] is False
        assert request_id == "phase3-c2-r1"
        assert timeout == 10
        return 200, {
            "generation_seconds": 0.5,
            "total_request_seconds": 0.6,
            "input_tokens": 10,
            "output_tokens": 16,
        }

    record = execute_request(make_config(), 2, 1, post=fake_post)

    assert record.success is True
    assert record.output_tokens == 16
    assert record.server_total_seconds == 0.6


@pytest.mark.parametrize(
    "config",
    [
        BenchmarkConfig("localhost", "x", 1, 1, (1,), 0, 1),
        BenchmarkConfig("http://localhost", " ", 1, 1, (1,), 0, 1),
        BenchmarkConfig("http://localhost", "x", 0, 1, (1,), 0, 1),
        BenchmarkConfig("http://localhost", "x", 1, 0, (1,), 0, 1),
        BenchmarkConfig("http://localhost", "x", 1, 1, (0,), 0, 1),
    ],
)
def test_invalid_benchmark_config_is_rejected(config: BenchmarkConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
