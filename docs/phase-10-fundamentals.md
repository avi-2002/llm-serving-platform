# Phase 10: Kubernetes orchestration

## What Kubernetes adds after Docker

Docker runs a container. Kubernetes continuously compares the declared desired
state with reality. If a Pod crashes, a controller creates a replacement. A
Service gives changing Pods a stable network address, and probes decide whether a
Pod should receive traffic or be restarted.

## Resources in this phase

- **Namespace:** isolates the learning deployment as `llm-serving`.
- **ConfigMap:** stores non-secret model and Ray settings outside the image.
- **PersistentVolumeClaim:** requests 2 GiB for the Hugging Face model cache.
- **Deployment:** declares one Pod running `ray-llm-api` from the Phase 9 image.
- **Service:** gives the Pod a stable internal address and port.
- **Kustomization:** applies all resources as one unit.

## Pod, container, and replica

A container is the running image. A Pod is Kubernetes' smallest scheduled unit
and can contain one or more containers. A Deployment manages a requested number
of interchangeable Pod replicas. Here one Pod contains one Ray head process,
HTTP ingress, and one model worker replica.

## The three probes

- The **startup probe** gives model download and loading up to five minutes. Until
  it succeeds, Kubernetes does not run the other probes.
- The **readiness probe** uses `/ready`. Failed readiness removes the Pod from
  Service traffic without restarting it.
- The **liveness probe** uses `/health`. Repeated failure tells Kubernetes to
  restart the container.

## Scheduling and limits

The Pod requests one CPU and 2 GiB RAM, which the scheduler uses for placement.
It may use up to four CPUs and 6 GiB RAM. CPU excess is throttled; exceeding the
memory limit can cause an out-of-memory termination and restart.

## Storage and security

The model cache is mounted from a persistent claim, while `/tmp` and `/dev/shm`
are temporary Pod storage. `fsGroup` lets the non-root process write the mounted
claim. The container drops Linux capabilities, prevents privilege escalation,
uses a read-only root filesystem, and retains writable mounts only where needed.

## Local cluster prerequisite

`kubectl` is installed, but Docker Desktop Kubernetes must be enabled before
deployment. In Docker Desktop, open Settings, select Kubernetes, enable it, apply
the change, and wait until the cluster reports running.

Verify:

```bash
kubectl config current-context
kubectl cluster-info
kubectl get nodes
```

The expected context is normally `docker-desktop`.

## Deploy

```bash
kubectl apply -k kubernetes
kubectl get pods -n llm-serving -w
```

The first start may download the model again because Kubernetes' persistent
volume is separate from the Compose named volume.

Inspect the system:

```bash
kubectl get all,pvc -n llm-serving
kubectl describe pod -n llm-serving -l app.kubernetes.io/name=llm-serving
kubectl logs -n llm-serving deployment/llm-serving -f
```

## Access the ClusterIP Service

ClusterIP is intentionally not exposed outside the cluster. Forward a temporary
local port:

```bash
kubectl port-forward -n llm-serving service/llm-serving 8080:80
```

In another terminal:

```bash
curl http://127.0.0.1:8080/ready
curl -X POST http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what a Kubernetes Pod is.","max_new_tokens":32}'
```

## Remove the learning deployment

```bash
kubectl delete -k kubernetes
```

Deleting the manifest also deletes its persistent volume claim. Do this only when
you intend to remove the Kubernetes model cache.

## Topics to study

- Control plane, nodes, kubelet, scheduler, and controllers.
- Declarative desired state and reconciliation loops.
- Pods, Deployments, ReplicaSets, Services, and EndpointSlices.
- ConfigMaps versus Secrets.
- Requests, limits, throttling, eviction, and OOMKilled.
- Persistent volumes, claims, access modes, and storage classes.
- Rolling updates versus the `Recreate` strategy.
- Horizontal Pod Autoscaling and why LLM scaling needs workload-aware metrics.

## Official reading

- Kubernetes concepts: <https://kubernetes.io/docs/concepts/>
- Startup, readiness, and liveness probes:
  <https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-probes>
- Container resource management:
  <https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/>
- Docker Desktop Kubernetes:
  <https://docs.docker.com/desktop/use-desktop/kubernetes/>
