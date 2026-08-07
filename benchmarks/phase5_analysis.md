# Phase 5 dynamic batching analysis

## Question

Can one model replica serve concurrent requests more efficiently by combining
them into a single tensor operation?

The comparison held the model, prompt, greedy 16-token output, one replica, four
PyTorch threads, and 12 measured requests per concurrency level constant. Only
the batching configuration changed: batch size 1 with no wait versus batch size
4 with a maximum 20 ms wait.

## Results

| Concurrency | Actual batch | Unbatched req/s | Batched req/s | Throughput change | Unbatched p95 | Batched p95 | p95 change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.87 | 0.90 | +3.3% | 1.34 s | 1.11 s | -17.1% |
| 2 | 2 | 0.83 | 1.00 | +21.3% | 2.60 s | 2.00 s | -23.1% |
| 4 | 4 | 0.75 | 1.93 | +156.8% | 9.03 s | 2.10 s | -76.8% |

All 72 measured requests succeeded. The batch telemetry proves that the candidate
formed batches of 2 at concurrency 2 and batches of 4 at concurrency 4; this was
not merely a configuration change.

At concurrency 4, the batched model call took about 2.06 seconds, compared with
about 1.16 seconds for a single unbatched call. That batched call completed four
16-token responses, however, so it did four requests' work in less than twice the
time. Aggregate output throughput consequently rose from 12.01 to 30.83 tokens/s.

## Interpretation

Dynamic batching is like putting four passengers into one bus instead of sending
four separate cars. The trip takes a little longer than one car trip, but carries
four times as many passengers. Here, PyTorch processed the batch dimension more
efficiently than four separate model invocations.

Concurrency 1 never formed a batch. Its small differences should be treated as
normal run-to-run noise, not evidence that batching improves isolated requests.
The candidate can also wait up to 20 ms for partners, which is a real low-traffic
latency cost even though it is small beside this model's roughly one-second
generation time.

The strongest result is at concurrency 4: throughput was 2.57x the control, p50
fell 49.7%, p95 fell 76.8%, and p99 fell 81.9%. Batching improved latency here
because it removed a much larger queueing delay than the 20 ms collection wait.

## What this does not prove

- Twelve requests per level is a learning experiment, not a production capacity
  study; longer repeated runs are needed for stable tail percentiles.
- All prompts and output limits matched. Mixed lengths can waste padded compute
  and cause head-of-line blocking.
- The result applies to this model and M1 CPU configuration. Batch size should be
  retuned on each target machine.
- Memory usage and quality under sampled decoding still need dedicated tests.

Raw repeatable-run artifacts remain in `work/phase5-unbatched.json` and
`work/phase5-batched-4.json`; the curated measurements are in
`benchmarks/phase5_batching_comparison.json`.
