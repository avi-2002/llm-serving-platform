# Phase 4 Ray Serve replica analysis

The same controlled benchmark was run against one and two fixed Ray Serve model
replicas. Each replica reserved four Ray CPUs, configured PyTorch for four compute
threads, accepted one ongoing generation, and loaded an independent copy of the
same model.

## Results

| Replicas | Concurrency | Requests/s | Tokens/s | p50 | p95 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.91 | 14.5 | 1.090 s | 1.141 s | 1.141 s | 0 |
| 1 | 2 | 0.90 | 14.4 | 2.199 s | 2.278 s | 2.284 s | 0 |
| 1 | 4 | 0.76 | 12.2 | 3.876 s | 7.725 s | 8.712 s | 0 |
| 2 | 1 | 0.86 | 13.8 | 1.096 s | 1.432 s | 1.553 s | 0 |
| 2 | 2 | 0.99 | 15.9 | 2.016 s | 2.043 s | 2.043 s | 0 |
| 2 | 4 | 0.96 | 15.3 | 4.003 s | 4.311 s | 4.312 s | 0 |

## Routing was correct

The one-replica run reported one distinct actor ID. The two-replica run reported
two distinct actor IDs, and each handled four of the eight requests at every
concurrency level. The small throughput gain was therefore not caused by routing
all traffic to one replica.

## Two replicas helped the tail under higher concurrency

At concurrency 4, compared with one Ray replica, two replicas:

- increased request throughput by about 25.1% (`0.76 -> 0.96 requests/s`);
- increased aggregate output throughput by about 25.1% (`12.2 -> 15.3 tokens/s`);
- reduced p95 latency by about 44.2% (`7.725 -> 4.311 seconds`);
- reduced p99 latency by about 50.5% (`8.712 -> 4.312 seconds`);
- reduced mean queue/overhead from 2.99 to 1.59 seconds;
- completed all requests without errors.

p50 did not improve at concurrency 4 because both replicas slowed under shared CPU
contention. The improvement appeared primarily in the slow tail and total service
capacity.

## Why throughput did not double

At concurrency 2, one replica generated a response in about 1.11 seconds. With two
replicas active together, each generation averaged about 2.01 seconds. Both model
actors executed concurrently, but they shared one physical M1 CPU containing four
performance and four efficiency cores, as well as memory bandwidth and caches.

Horizontal scaling on one machine does not create new hardware. Two processes
divided the existing compute, so the expected ideal 2x throughput increase was
mostly offset by slower per-replica execution. Throughput increased about 10.6% at
concurrency 2 rather than 100%.

## Comparison with the Phase 3 server

At concurrency 4, the original Phase 3 server achieved 0.87 requests/s with p95
4.72 seconds. Two Ray replicas achieved 0.96 requests/s with p95 4.31 seconds:
about 10.3% higher throughput and 8.7% lower p95 in these small runs.

Ray's larger value is architectural rather than a dramatic laptop speedup. The
same deployment abstraction can place replicas on separate nodes or GPUs, where
each replica can receive genuinely additional hardware instead of competing for
one M1 CPU.

## Limitations

- Eight requests per level are insufficient for production percentile claims.
- CPU thread topology includes performance and efficiency cores with unequal speed.
- Each replica duplicates model weights and increases memory consumption.
- Only one short prompt and 16-token output were tested.
- Fixed replicas were tested; no autoscaling or batching was enabled.
- Local loopback excludes real network and multi-node communication costs.

The defensible conclusion is not “Ray makes the model twice as fast.” It is:

> Ray successfully routed work across isolated model replicas. On this constrained
> single machine, two replicas modestly increased throughput and materially reduced
> high-concurrency tail latency, while shared CPU contention limited scaling.
