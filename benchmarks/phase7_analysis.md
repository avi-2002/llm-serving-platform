# Phase 7 observability verification

## What the metrics proved

The API recorded one successful normal request and one successful streaming
request, with no observed error. Both in-progress gauges returned to zero after
completion, showing that the service did not report work as permanently stuck.

| Signal | Normal request | Streaming request |
|---|---:|---:|
| Completed requests | 1 | 1 |
| Total duration | 4.158 s | 4.224 s |
| Time to first displayed chunk | Not applicable | 0.257 s |
| Output measurement | 64 tokens | 342 characters |

The two total durations were close, reinforcing the Phase 6 lesson: streaming
changes when output becomes visible rather than substantially changing total
model compute. The streamed response began after about 6.1% of its total duration,
or roughly 3.97 seconds before completion.

The batch-size histogram recorded one batch of size one. Only the normal endpoint
reports model-token and batch telemetry in this standard API implementation; the
streaming endpoint reports decoded characters and first-chunk latency instead.

## Why the zero gauge matters

`llm_requests_in_progress` rises when inference starts and falls when it finishes.
A value of zero after both commands means no request was executing at scrape time.
If this value remained elevated without traffic, it could indicate a slow, hung,
or incorrectly cleaned-up operation.

## Limits

This verifies instrumentation correctness with two requests; it is not a load
test. Useful alerts and percentile dashboards require Prometheus to scrape many
observations over time. Phase 3 and Phase 5 remain the controlled performance
experiments.
