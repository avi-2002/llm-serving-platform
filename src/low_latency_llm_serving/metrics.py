"""Prometheus metrics for inference behavior and service health."""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
BATCH_BUCKETS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class ServingMetrics:
    """Application-scoped metrics that avoid global-registry test collisions."""

    registry: CollectorRegistry
    requests: Counter
    in_progress: Gauge
    request_duration: Histogram
    first_chunk: Histogram
    output_tokens: Counter
    output_characters: Counter
    batch_size: Histogram

    @classmethod
    def create(cls) -> "ServingMetrics":
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests=Counter(
                "llm_requests_total",
                "Completed LLM inference requests.",
                ("endpoint", "outcome"),
                registry=registry,
            ),
            in_progress=Gauge(
                "llm_requests_in_progress",
                "LLM inference requests currently executing.",
                ("endpoint",),
                registry=registry,
            ),
            request_duration=Histogram(
                "llm_request_duration_seconds",
                "End-to-end inference request duration.",
                ("endpoint",),
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
            first_chunk=Histogram(
                "llm_time_to_first_chunk_seconds",
                "Time until the first displayed streaming text chunk.",
                buckets=LATENCY_BUCKETS,
                registry=registry,
            ),
            output_tokens=Counter(
                "llm_output_tokens_total",
                "Model output tokens returned by completed requests.",
                ("endpoint",),
                registry=registry,
            ),
            output_characters=Counter(
                "llm_stream_output_characters_total",
                "Decoded characters delivered through streaming.",
                registry=registry,
            ),
            batch_size=Histogram(
                "llm_batch_size",
                "Observed number of requests in a model batch.",
                buckets=BATCH_BUCKETS,
                registry=registry,
            ),
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
