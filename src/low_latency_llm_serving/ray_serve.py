"""Ray Serve deployment that preserves the Phase 2 HTTP contract."""

from __future__ import annotations

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


@serve.deployment(name="LocalLLMDeployment")
@serve.ingress()
class RayLLMDeployment:
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

    def __serve_build_asgi_app__(self) -> FastAPI:
        """Build FastAPI inside the replica to avoid serializing its thread locks."""
        application = FastAPI(
            title="Ray Serve LLM API",
            version="0.4.0",
            description="Phase 4: replica-based local LLM inference with Ray Serve.",
        )

        @application.get(
            "/health", response_model=HealthResponse, tags=["operations"]
        )
        async def health_endpoint() -> HealthResponse:
            return await self.health()

        @application.get("/ready", response_model=ReadyResponse, tags=["operations"])
        async def ready_endpoint() -> ReadyResponse:
            return await self.ready()

        @application.get(
            "/v1/metadata", response_model=MetadataResponse, tags=["inference"]
        )
        async def metadata_endpoint() -> MetadataResponse:
            return await self.metadata()

        @application.post(
            "/v1/generate", response_model=GenerateResponse, tags=["inference"]
        )
        async def generate_endpoint(
            payload: GenerateRequest, request: Request
        ) -> GenerateResponse:
            return await self.generate(payload, request)

        return application

    async def health(self) -> HealthResponse:
        return HealthResponse(model_status="ready")

    async def ready(self) -> ReadyResponse:
        return ReadyResponse(model_id=self.model.model_id)

    async def metadata(self) -> MetadataResponse:
        return MetadataResponse(
            model_id=self.model.model_id,
            device=self.model.device.type,
            dtype=str(self.model.dtype).removeprefix("torch."),
            load_seconds=self.model.load_seconds,
        )

    async def generate(
        self, payload: GenerateRequest, request: Request
    ) -> GenerateResponse:
        request_started = perf_counter()
        settings = GenerationSettings(
            max_new_tokens=payload.max_new_tokens,
            do_sample=payload.do_sample,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
        )
        result = await asyncio.to_thread(self.model.generate, payload.prompt, settings)
        return GenerateResponse(
            request_id=request.headers.get("x-request-id", str(uuid4())),
            replica_id=self.replica_id,
            model_id=result.model_id,
            device=result.device,
            dtype=result.dtype,
            response=result.response,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_seconds=result.generation_seconds,
            total_request_seconds=perf_counter() - request_started,
            tokens_per_second=result.tokens_per_second,
            parameters=GenerationParameters(**asdict(result.settings)),
        )


def build_application(config: RayServeConfig | None = None) -> serve.Application:
    config = config or config_from_environment()
    config.validate()
    return RayLLMDeployment.options(
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
    serve.run(build_application(config), blocking=True, route_prefix="/")


if __name__ == "__main__":
    main()
