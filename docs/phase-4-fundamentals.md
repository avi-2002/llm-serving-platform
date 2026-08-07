# Phase 4: Ray Serve fundamentals

Phase 3 showed that one protected model instance forms a queue under concurrent
traffic. Phase 4 replaces the manually managed model lock with Ray Serve's
deployment, replica, routing, and resource abstractions while preserving the same
HTTP contract and benchmark client.

## Ray Core and Ray Serve

Ray Core manages processes and resources:

- a local control plane tracks nodes, actors, and available CPUs;
- an actor is a stateful worker process;
- resource declarations tell Ray where work is allowed to run.

Ray Serve adds online request serving:

- a deployment describes a service component;
- a replica is one running copy of that deployment;
- an HTTP proxy accepts requests;
- a router assigns requests to available replicas;
- backpressure limits how much work can be queued or assigned.

Our deployment actor loads one independent `LocalLLM`. Two replicas therefore
mean two Python processes and two copies of the model weights.

## Request flow

```text
HTTP client
  -> Ray Serve HTTP proxy
  -> lightweight FastAPI ingress replica
  -> deployment handle and request router
  -> available ModelWorker replica
  -> LocalLLM generation
  -> JSON response with replica ID
```

The Phase 2 and Ray Serve APIs expose the same paths and request schema. This is
why the Phase 3 benchmark can compare them without special cases.

## Replicas are capacity, not free speed

One replica can perform one generation at a time because
`max_ongoing_requests=1`. With two replicas, Ray may perform two generations at
once:

```text
request A -> replica 1
request B -> replica 2
request C -> waits for either replica
```

Each response contains `replica_id`. The benchmark counts distinct IDs, providing
evidence that traffic reached multiple actors.

Replicas consume memory and CPU. On an 8-core, 16 GB M1, two replicas each reserve
four logical CPUs and configure PyTorch for four compute threads. More replicas
would oversubscribe CPU and duplicate model memory, so more is not automatically
better.

## Scheduling resources versus actual compute threads

`ray_actor_options={"num_cpus": 4}` tells Ray's scheduler that the replica requires
four CPUs. It does not by itself force PyTorch to use exactly four threads.

The deployment also calls `torch.set_num_threads(4)`. Keeping both values aligned
prevents each replica from independently trying to use every CPU core.

## `max_ongoing_requests`

This setting limits how many incomplete requests Ray assigns to each replica. We
set it to one because the underlying model call is a heavy, blocking generation
and we want concurrency to come from distinct model replicas rather than several
threads contending inside one replica.

Later batching work will deliberately allow multiple requests to be combined in
a controlled vectorized operation.

## Queue boundaries and timing

Ray's router may hold a request before the replica method starts. The response's
`total_request_seconds` begins inside the replica, so it does not include all Ray
proxy/router queueing. The benchmark's client latency does include that time.

For Ray experiments, this difference is expected:

```text
client latency - replica total time
  approximately equals proxy/router queueing + network/serialization overhead
```

## Fixed replicas before autoscaling

This phase uses one and two fixed replicas. Fixed capacity makes cause and effect
easy to observe. Autoscaling adds cold-start delay, decision intervals, scale-up
thresholds, and scale-down behavior; it will be tested only after fixed-replica
behavior is understood.

## FastAPI factory lesson

The current FastAPI object contains an internal thread lock and cannot be
serialized by Ray's `cloudpickle`. A lightweight ingress replica therefore builds
FastAPI locally from a factory and delegates inference to separate model workers
through a Ray deployment handle. This is why serialization boundaries matter in
distributed systems: objects that work within one process are not necessarily
transferable to another.

## Expected experiment

Run the same controlled benchmark against:

1. one Ray replica with four CPU threads;
2. two Ray replicas with four CPU threads each.

Compare throughput, p50/p95/p99, errors, and `replicas_observed`. Because both
replicas share one physical M1 CPU, two-replica throughput may improve, stay flat,
or even decline depending on memory bandwidth and compute contention. The result
must be measured rather than assumed.

## Topics to study

- Ray actors, tasks, object store, and logical resources.
- Ray Serve deployments, replicas, proxy, and request router.
- Backpressure and queue limits.
- Horizontal versus vertical scaling.
- CPU oversubscription and context switching.
- Process isolation and serialization with `cloudpickle`.
- Fixed replicas versus request-driven autoscaling.

## Official reading

- Ray Serve HTTP and FastAPI:
  <https://docs.ray.io/en/latest/serve/http-guide.html>
- Deployment configuration:
  <https://docs.ray.io/en/latest/serve/configure-serve-deployment.html>
- Ray Serve autoscaling:
  <https://docs.ray.io/en/latest/serve/autoscaling-guide.html>
