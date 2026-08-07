"""Concurrent HTTP benchmark for the local generation API."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    prompt: str
    max_new_tokens: int
    requests_per_level: int
    concurrency_levels: tuple[int, ...]
    warmup_requests: int
    timeout_seconds: float
    seed: int = 42

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.prompt.strip():
            raise ValueError("prompt must contain text")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if self.requests_per_level < 1:
            raise ValueError("requests_per_level must be positive")
        if not self.concurrency_levels or any(
            value < 1 for value in self.concurrency_levels
        ):
            raise ValueError("all concurrency levels must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    concurrency: int
    success: bool
    status_code: int | None
    client_latency_seconds: float
    server_generation_seconds: float | None
    server_total_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None
    replica_id: str | None = None


@dataclass(frozen=True)
class LevelSummary:
    concurrency: int
    attempted_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    wall_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    client_latency_mean_seconds: float | None
    client_latency_p50_seconds: float | None
    client_latency_p95_seconds: float | None
    client_latency_p99_seconds: float | None
    server_generation_mean_seconds: float | None
    server_total_mean_seconds: float | None
    server_overhead_mean_seconds: float | None
    replicas_observed: int


PostFunction = Callable[[str, dict[str, object], str, float], tuple[int, dict[str, Any]]]


def percentile(values: Sequence[float], percentage: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_level(
    records: Sequence[RequestRecord], concurrency: int, wall_seconds: float
) -> LevelSummary:
    successful = [record for record in records if record.success]
    latencies = [record.client_latency_seconds for record in successful]
    generation_times = [
        record.server_generation_seconds
        for record in successful
        if record.server_generation_seconds is not None
    ]
    server_totals = [
        record.server_total_seconds
        for record in successful
        if record.server_total_seconds is not None
    ]
    overheads = [
        record.server_total_seconds - record.server_generation_seconds
        for record in successful
        if record.server_total_seconds is not None
        and record.server_generation_seconds is not None
    ]
    output_tokens = sum(record.output_tokens or 0 for record in successful)
    failed_count = len(records) - len(successful)
    replica_ids = {record.replica_id for record in successful if record.replica_id}

    return LevelSummary(
        concurrency=concurrency,
        attempted_requests=len(records),
        successful_requests=len(successful),
        failed_requests=failed_count,
        error_rate=failed_count / len(records) if records else 0.0,
        wall_seconds=wall_seconds,
        requests_per_second=len(successful) / wall_seconds if wall_seconds else 0.0,
        output_tokens_per_second=output_tokens / wall_seconds if wall_seconds else 0.0,
        client_latency_mean_seconds=_mean(latencies),
        client_latency_p50_seconds=percentile(latencies, 50) if latencies else None,
        client_latency_p95_seconds=percentile(latencies, 95) if latencies else None,
        client_latency_p99_seconds=percentile(latencies, 99) if latencies else None,
        server_generation_mean_seconds=_mean(generation_times),
        server_total_mean_seconds=_mean(server_totals),
        server_overhead_mean_seconds=_mean(overheads),
        replicas_observed=len(replica_ids),
    )


def _get_json(url: str, timeout_seconds: float) -> tuple[int, dict[str, Any]]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return response.status, json.loads(response.read())


def _post_json(
    url: str,
    payload: dict[str, object],
    request_id: str,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.status, json.loads(response.read())


def wait_until_ready(
    base_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    deadline = perf_counter() + timeout_seconds
    last_error = "no response"
    while perf_counter() < deadline:
        try:
            status_code, body = _get_json(
                f"{base_url.rstrip('/')}/ready", timeout_seconds=2
            )
            if status_code == 200:
                return body
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"API did not become ready: {last_error}")


def fetch_metadata(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    status_code, body = _get_json(
        f"{base_url.rstrip('/')}/v1/metadata", timeout_seconds
    )
    if status_code != 200:
        raise RuntimeError(f"metadata request returned HTTP {status_code}")
    return body


def execute_request(
    config: BenchmarkConfig,
    concurrency: int,
    request_number: int,
    post: PostFunction = _post_json,
) -> RequestRecord:
    request_id = f"phase3-c{concurrency}-r{request_number}"
    payload: dict[str, object] = {
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "seed": config.seed,
    }
    started = perf_counter()
    try:
        status_code, body = post(
            f"{config.base_url.rstrip('/')}/v1/generate",
            payload,
            request_id,
            config.timeout_seconds,
        )
        latency = perf_counter() - started
        if status_code != 200:
            return RequestRecord(
                request_id=request_id,
                concurrency=concurrency,
                success=False,
                status_code=status_code,
                client_latency_seconds=latency,
                server_generation_seconds=None,
                server_total_seconds=None,
                input_tokens=None,
                output_tokens=None,
                error=str(body),
            )
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            success=True,
            status_code=status_code,
            client_latency_seconds=latency,
            server_generation_seconds=float(body["generation_seconds"]),
            server_total_seconds=float(body["total_request_seconds"]),
            input_tokens=int(body["input_tokens"]),
            output_tokens=int(body["output_tokens"]),
            error=None,
            replica_id=body.get("replica_id"),
        )
    except HTTPError as exc:
        latency = perf_counter() - started
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            success=False,
            status_code=exc.code,
            client_latency_seconds=latency,
            server_generation_seconds=None,
            server_total_seconds=None,
            input_tokens=None,
            output_tokens=None,
            error=exc.read().decode(errors="replace"),
        )
    except (URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        latency = perf_counter() - started
        return RequestRecord(
            request_id=request_id,
            concurrency=concurrency,
            success=False,
            status_code=None,
            client_latency_seconds=latency,
            server_generation_seconds=None,
            server_total_seconds=None,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_level(
    config: BenchmarkConfig,
    concurrency: int,
    post: PostFunction = _post_json,
) -> tuple[LevelSummary, list[RequestRecord]]:
    started = perf_counter()
    records: list[RequestRecord] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(execute_request, config, concurrency, index, post)
            for index in range(1, config.requests_per_level + 1)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    wall_seconds = perf_counter() - started
    records.sort(key=lambda record: record.request_id)
    return summarize_level(records, concurrency, wall_seconds), records


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    config.validate()
    wait_until_ready(config.base_url, config.timeout_seconds)
    metadata = fetch_metadata(config.base_url, config.timeout_seconds)

    for index in range(1, config.warmup_requests + 1):
        warmup = execute_request(config, concurrency=1, request_number=-index)
        if not warmup.success:
            raise RuntimeError(f"warmup request failed: {warmup.error}")

    levels: list[dict[str, object]] = []
    for concurrency in config.concurrency_levels:
        summary, records = run_level(config, concurrency)
        levels.append(
            {
                "summary": asdict(summary),
                "requests": [asdict(record) for record in records],
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "client_environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "model_metadata": metadata,
        "config": asdict(config),
        "levels": levels,
    }


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_summary(result: dict[str, object]) -> None:
    print("\nPhase 3 benchmark summary")
    print("concurrency  success  req/s  tok/s  p50(s)  p95(s)  p99(s)  errors")
    for level in result["levels"]:  # type: ignore[index]
        summary = level["summary"]  # type: ignore[index]
        print(
            f"{summary['concurrency']:>11}  "
            f"{summary['successful_requests']:>7}/"
            f"{summary['attempted_requests']:<3}  "
            f"{summary['requests_per_second']:>5.2f}  "
            f"{summary['output_tokens_per_second']:>5.1f}  "
            f"{_format_optional(summary['client_latency_p50_seconds']):>6}  "
            f"{_format_optional(summary['client_latency_p95_seconds']):>6}  "
            f"{_format_optional(summary['client_latency_p99_seconds']):>6}  "
            f"{summary['failed_requests']:>6}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure repeated and concurrent requests to the local LLM API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--prompt",
        default="Explain the difference between latency and throughput in two sentences.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=5, dest="requests_per_level")
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4]
    )
    parser.add_argument("--warmup", type=int, default=1, dest="warmup_requests")
    parser.add_argument("--timeout", type=float, default=120, dest="timeout_seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        base_url=args.base_url,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        requests_per_level=args.requests_per_level,
        concurrency_levels=tuple(args.concurrency),
        warmup_requests=args.warmup_requests,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )
    result = run_benchmark(config)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nRaw results: {args.output}")


if __name__ == "__main__":
    main()
