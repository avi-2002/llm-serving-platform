# Phase 10 Kubernetes deployment analysis

## Verified deployment

Docker Desktop Kubernetes v1.36.1 accepted the Kustomize resources and reconciled
one Ray Serve Pod on `desktop-control-plane`. The local-path provisioner created
and bound the requested 2 GiB persistent volume. The final Pod ran as UID/GID
10001, reached Ready status, and had zero restarts.

Through a temporary port-forward to the ClusterIP Service, `/ready` reported Qwen
available and a real request returned 23 output tokens. Model generation took
5.55 seconds and the complete request took 5.61 seconds. The Ray replica ID and
Prometheus success/batch observations confirmed that traffic passed through Ray
Serve rather than bypassing the deployed service.

## Bugs found only by live deployment

The first Pod exposed two configuration problems that offline YAML tests could
not detect:

1. Ray received only 512 MiB of shared memory and fell back to slower `/tmp`.
   The memory-backed `/dev/shm` volume was increased to 2 GiB.
2. Ray's proxy listened on loopback, so in-process calls worked while kubelet
   probes to the Pod IP received connection refused. Ray now explicitly binds to
   `0.0.0.0:8000`.

The first replacement still used Kubernetes' cached mutable `phase9` tag. The
fixed build was retagged `phase10`, demonstrating why deployments should use
unique version tags or immutable image digests.

## Kubernetes behaviors observed

- The Deployment controller created ReplicaSets and replacement Pods.
- The scheduler assigned the Pod only after considering its resource request.
- Dynamic volume provisioning changed the claim from Pending to Bound.
- The startup probe withheld readiness during Ray/model initialization.
- The Service EndpointSlice changed to ready only after the Pod passed probes.
- The persistent claim retained the 954 MB model cache across Pod replacements.
- The Recreate strategy avoided running two memory-heavy model Pods together.

This proves a working single-node local deployment. It does not yet prove
multi-node storage, cloud load balancing, registry-based image distribution,
autoscaling, high availability, or production security operations.
