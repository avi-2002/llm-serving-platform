# Phase 3: Measurement and load-testing fundamentals

## Latency and throughput answer different questions

- **Latency:** how long one request waits from client send to client response.
- **Throughput:** how much useful work the complete service finishes per unit time.

The same system can have stable throughput and terrible latency. That happened in
our concurrency test: the one-model worker stayed busy, but new requests waited
longer in line.

## Why averages are insufficient

An average hides the shape of the latency distribution. If nine requests take one
second and one takes ten seconds, the average is 1.9 seconds, which describes
neither the common experience nor the worst experience well.

- `p50` is the median: half of observations are at or below it.
- `p95` is the value at or below which roughly 95% of observations fall.
- `p99` focuses further into the slow tail.

Reliable tail percentiles require many samples. With only eight observations,
Phase 3's p95 and p99 are learning indicators, not service-level evidence.

## Client time versus server time

The harness records client wall-clock latency. The server separately returns:

- model generation time;
- total time inside the API endpoint.

Conceptually:

```text
client latency
  = network and client overhead
  + server queueing
  + model generation
  + response serialization
```

Because the test uses local loopback, network cost is tiny. In production it may
not be.

## Closed-loop concurrency

The harness uses a fixed worker pool. At concurrency 4, four requests can be in
flight; when one finishes, that worker starts its next request until the configured
request count is complete. This is called a closed-loop test.

Closed-loop tests are easy to reproduce, but they can understate overload because
slow responses automatically reduce the rate at which the client sends new work.
An open-loop test instead sends requests according to an arrival schedule even
when the server is slow.

## Warm-up and cold-start effects

The unmeasured warm-up request ensures weights are loaded and exercises lazy
runtime paths before measurement. Without warm-up, the first observation can
include cache population or one-time initialization and distort the distribution.

Cold-start latency is still important, but it must be measured as a separate
experiment rather than mixed into steady-state latency.

## Controlled experiments

A useful performance comparison changes one independent variable at a time. This
phase held constant:

- model and software versions;
- CPU and FP32 precision;
- prompt text;
- greedy decoding and seed;
- maximum/output token count;
- machine and local network path.

Concurrency was the independent variable. Client latency, throughput, errors, and
server timing were dependent measurements.

## Queueing lesson from this phase

At concurrency 1, non-generation endpoint time was about 1 millisecond. At
concurrency 4 it averaged 2.57 seconds, even though generation itself was slightly
faster. The additional time was primarily queue waiting behind the generation
lock.

This is why “the model takes about 1.2 seconds” does not imply “users receive an
answer in about 1.2 seconds” under load.

## Exercises

1. Draw the request queue for four simultaneous requests and one model worker.
2. Explain why throughput barely increased while p50 latency nearly quadrupled.
3. Rerun with 16 versus 64 output tokens and compare saturation throughput.
4. Increase samples to at least 30 per level and observe how percentiles stabilize.
5. Introduce a deliberately tiny timeout and inspect error-rate reporting.
6. Explain why client latency must be greater than or approximately equal to
   server endpoint time for the same request.

## Topics to study

- Little's Law: average items in a system equal arrival rate times average time.
- Queueing delay and service time.
- Closed-loop versus open-loop load generation.
- Coordinated omission in load tests.
- Service-level indicators (SLIs), objectives (SLOs), and error budgets.
- Histograms versus summaries for production latency metrics.
