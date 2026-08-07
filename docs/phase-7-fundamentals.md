# Phase 7: Service observability with Prometheus metrics

## Why this phase exists

Benchmarks answer a question during a controlled experiment. Observability tells
us what the service is doing continuously: whether traffic increased, requests
failed, latency degraded, batching stopped working, or users waited too long for
streamed text.

The API now exposes Prometheus text data at `GET /metrics`.

## The three metric types we use

- **Counter:** only increases. We use counters for completed requests, output
  tokens, and streamed characters.
- **Gauge:** moves up and down. We use a gauge for requests currently executing.
- **Histogram:** places observations into fixed buckets. We use histograms for
  request latency, time to first streamed chunk, and batch size. Prometheus can
  calculate rates and fleet-wide percentiles from these buckets.

## Metrics added

| Metric | Meaning |
|---|---|
| `llm_requests_total` | Completed requests split by endpoint and outcome |
| `llm_requests_in_progress` | Requests executing right now |
| `llm_request_duration_seconds` | Complete inference latency distribution |
| `llm_time_to_first_chunk_seconds` | Streaming perceived-latency distribution |
| `llm_output_tokens_total` | Output tokens from non-streaming requests |
| `llm_stream_output_characters_total` | Text characters delivered while streaming |
| `llm_batch_size` | Distribution of observed model batch sizes |

## Why request IDs are not labels

Every unique label combination creates another time series. A request ID, prompt,
or user ID can create millions of series and consume large amounts of memory.
Metrics use only bounded labels (`endpoint` and `outcome`). Request-specific data
belongs in logs or traces.

## Run the experiment

Start the standard API as in Phase 6, make one normal request and one streaming
request, then inspect:

```bash
curl -s http://127.0.0.1:8000/metrics | grep '^llm_'
```

Look for request counts, duration buckets, the first-chunk observation, and an
in-progress value that has returned to zero.

## Observed result

The verification run recorded one successful normal request and one successful
streaming request. Their total durations were 4.158 and 4.224 seconds. Streaming
produced its first displayed text in 0.257 seconds, about 6.1% into the request,
while both in-progress gauges correctly returned to zero.

The normal request recorded 64 output tokens and a batch size of one. The streamed
request recorded 342 delivered characters. See `benchmarks/phase7_analysis.md`
for the interpretation and limits.

## Histograms in plain language

A latency histogram keeps several baskets such as "under 0.5 seconds", "under 1
second", and "under 5 seconds". Every request goes into all baskets whose upper
limit it fits. Prometheus uses the accumulated basket counts to estimate p50,
p95, and p99 across replicas.

## Topics to study

- The four golden signals: latency, traffic, errors, and saturation.
- Counters, gauges, histograms, and summaries.
- PromQL `rate()` and `histogram_quantile()`.
- Metric cardinality and label design.
- Pull-based scraping and service discovery.
- Grafana dashboards and alert thresholds.

## Official reading

- Prometheus Python instrumentation:
  <https://prometheus.github.io/client_python/instrumenting/>
- Prometheus Python ASGI export:
  <https://prometheus.github.io/client_python/exporting/http/asgi/>
- Prometheus metric types:
  <https://prometheus.io/docs/concepts/metric_types/>
