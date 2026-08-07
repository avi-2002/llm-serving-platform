# Phase 3 concurrency baseline analysis

The Phase 2 API was tested with one warm-up followed by eight measured requests
at each concurrency level. Model, prompt, output length, decoding strategy, seed,
and device were fixed. Only the number of simultaneous client workers changed.

| Concurrency | Success | Requests/s | Output tokens/s | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 0.83 | 13.3 | 1.175 s | 1.322 s | 1.368 s |
| 2 | 8/8 | 0.86 | 13.7 | 2.325 s | 2.382 s | 2.391 s |
| 4 | 8/8 | 0.87 | 13.9 | 4.505 s | 4.720 s | 4.730 s |

## Conclusion

The service is saturated by one active generation. Moving from concurrency 1 to
4 increased request throughput by only about 4.1%, from 0.83 to 0.87 requests/s,
while p50 latency grew about 3.83x, from 1.175 to 4.505 seconds.

This is the expected consequence of the Phase 2 generation lock. Concurrent
HTTP requests are accepted, but only one request at a time uses the model. Other
requests wait in the application queue.

The server timing makes the queue visible:

| Concurrency | Mean generation | Mean endpoint total | Mean non-generation time |
|---:|---:|---:|---:|
| 1 | 1.198 s | 1.199 s | 0.001 s |
| 2 | 1.165 s | 2.184 s | 1.019 s |
| 4 | 1.152 s | 3.720 s | 2.568 s |

The model's own generation time stayed around 1.15-1.20 seconds. The growing
latency came primarily from waiting, not slower neural-network computation.

## What this baseline proves

- All 24 measured requests succeeded, so error rate was 0%.
- The API can accept concurrent connections without corrupting model execution.
- A single serialized model does not gain meaningful throughput from more clients.
- Queueing causes tail latency to rise before throughput improves.
- Future serving work needs replicas, batching, or both—not merely more async code.

## Limitations

- Eight samples per level are too few for a production p95 or p99 claim. The
  interpolated percentiles only demonstrate the calculation and observed trend.
- Only one prompt length and one 16-token output length were tested.
- Tests ran on local loopback, so they exclude real network latency.
- CPU FP32 was tested; MPS, larger models, and quantized models may differ.
- Concurrency was closed-loop: each worker starts another request only after its
  previous request completes.
- Results represent this machine and locked software environment, not universal
  Qwen performance.

Later phases should use more samples, multiple prompt/output-length groups, and
an open-loop arrival-rate test when production capacity planning begins.
