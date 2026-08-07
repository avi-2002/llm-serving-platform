"""Ray Serve application with separate HTTP ingress and model replicas."""

import asyncio
import os
from dataclasses import asdict, dataclass
from time import perf_counter
from uuid import uuid4

import ray
import torch
from fastapi import FastAPI, Request
from ray import serve

from low_latency_llm_serving.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationParameters,
    HealthResponse,
    MetadataResponse,
    ReadyResponse,
)
from low_latency_llm_serving.inference import (
    DEFAULT_MODEL_ID,
    GenerationSettings,
    LocalLLM,
)

MODEL_WORKER_NAME = "ModelWorker"
APPLICATION_NAME = "default"


@dataclass(frozen=True)
class RayServeConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    dtype: str = "float32"
    num_replicas: int = 1
    cpus_per_replica: float = 4
    torch_threads_per_replica: int = 4
    max_queued_requests: int = 100

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must contain text")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be float32 or float16")
        if not 1 <= self.num_replicas <= 8:
            raise ValueError("num_replicas must be between 1 and 8")
        if self.cpus_per_replica <= 0:
            raise ValueError("cpus_per_replica must be positive")
        if self.torch_threads_per_replica < 1:
            raise ValueError("torch_threads_per_replica must be positive")
        if self.max_queued_requests < 0:
            raise ValueError("max_queued_requests cannot be negative")


def config_from_environment() -> RayServeConfig:
    return RayServeConfig(
        model_id=os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID),
        device=os.getenv("LLM_DEVICE", "cpu"),
        dtype=os.getenv("LLM_DTYPE", "float32"),
        num_replicas=int(os.getenv("RAY_NUM_REPLICAS", "1")),
        cpus_per_replica=float(os.getenv("RAY_CPUS_PER_REPLICA", "4")),
        torch_threads_per_replica=int(os.getenv("TORCH_THREADS_PER_REPLICA", "4")),
        max_queued_requests=int(os.getenv("RAY_MAX_QUEUED_REQUESTS", "100")),
    )


@serve.deployment(name=MODEL_WORKER_NAME)
class ModelWorker:
    """One stateful Ray actor containing one independently loaded model."""

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        torch_threads: int,
    ) -> None:
        if device == "cpu":
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(1)
        self.model = LocalLLM(model_id=model_id, device=device, dtype=dtype)
        self.replica_id = str(ray.get_runtime_context().get_actor_id())

    async def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model.model_id,
            "device": self.model.device.type,
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "load_seconds": self.model.load_seconds,
        }

    async def generate(self, payload: dict[str, object]) -> dict[str, object]:
        settings = GenerationSettings(
            max_new_tokens=int(payload["max_new_tokens"]),
            do_sample=bool(payload["do_sample"]),
            temperature=float(payload["temperature"]),
            top_p=float(payload["top_p"]),
            seed=int(payload["seed"]),
        )
        result = await asyncio.to_thread(
            self.model.generate, str(payload["prompt"]), settings
        )
        return {
            "replica_id": self.replica_id,
            "model_id": result.model_id,
            "device": result.device,
            "dtype": result.dtype,
            "response": result.response,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "generation_seconds": result.generation_seconds,
            "tokens_per_second": result.tokens_per_second,
            "settings": asdict(result.settings),
        }


def build_ingress_app() -> FastAPI:
    """Construct FastAPI inside the ingress replica, as recommended by Ray."""
    application = FastAPI(
        title="Ray Serve LLM API",
        version="0.4.0",
        description="Phase 4: replica-based local LLM inference with Ray Serve.",
    )

    def model_handle():
        return serve.get_deployment_handle(MODEL_WORKER_NAME, app_name=APPLICATION_NAME)

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(model_status="ready")

    @application.get("/ready", response_model=ReadyResponse, tags=["operations"])
    async def ready() -> ReadyResponse:
        metadata = await model_handle().metadata.remote()
        return ReadyResponse(model_id=str(metadata["model_id"]))

    @application.get(
        "/v1/metadata", response_model=MetadataResponse, tags=["inference"]
    )
    async def metadata() -> MetadataResponse:
        result = await model_handle().metadata.remote()
        return MetadataResponse.model_validate(result)

    @application.post(
        "/v1/generate", response_model=GenerateResponse, tags=["inference"]
    )
    async def generate(
        payload: GenerateRequest, request: Request
    ) -> GenerateResponse:
        request_started = perf_counter()
        result = await model_handle().generate.remote(payload.model_dump())
        return GenerateResponse(
            request_id=request.headers.get("x-request-id", str(uuid4())),
            replica_id=str(result["replica_id"]),
            model_id=str(result["model_id"]),
            device=str(result["device"]),
            dtype=str(result["dtype"]),
            response=str(result["response"]),
            input_tokens=int(result["input_tokens"]),
            output_tokens=int(result["output_tokens"]),
            generation_seconds=float(result["generation_seconds"]),
            total_request_seconds=perf_counter() - request_started,
            tokens_per_second=float(result["tokens_per_second"]),
            parameters=GenerationParameters.model_validate(result["settings"]),
        )

    return application


@serve.deployment(name="HTTPIngress", ray_actor_options={"num_cpus": 0})
@serve.ingress(build_ingress_app)
class HTTPIngress:
    """Lightweight HTTP boundary; model work is delegated through a handle."""

    def __init__(self, model_worker) -> None:
        # Binding the child ensures Ray deploys it as part of this application.
        self._model_worker = model_worker


def build_application(config: RayServeConfig | None = None) -> serve.Application:
    config = config or config_from_environment()
    config.validate()
    worker = ModelWorker.options(
        num_replicas=config.num_replicas,
        max_ongoing_requests=1,
        max_queued_requests=config.max_queued_requests,
        ray_actor_options={"num_cpus": config.cpus_per_replica},
    ).bind(
        config.model_id,
        config.device,
        config.dtype,
        config.torch_threads_per_replica,
    )
    return HTTPIngress.bind(worker)


application = build_application()


def main() -> None:
    config = config_from_environment()
    config.validate()
    ray.init(
        address="local",
        include_dashboard=False,
        num_cpus=int(os.getenv("RAY_TOTAL_CPUS", "8")),
        log_to_driver=True,
    )
    serve.run(
        build_application(config),
        blocking=True,
        name=APPLICATION_NAME,
        route_prefix="/",
    )


if __name__ == "__main__":
    main()
