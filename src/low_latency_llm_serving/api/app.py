"""FastAPI application and endpoint definitions."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from low_latency_llm_serving.api.runtime import ModelRuntime, RuntimeConfig
from low_latency_llm_serving.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationParameters,
    HealthResponse,
    MetadataResponse,
    ReadyResponse,
)
from low_latency_llm_serving.inference import DEFAULT_MODEL_ID, GenerationSettings
from low_latency_llm_serving.metrics import ServingMetrics


def _display_dtype(value: object) -> str:
    return str(value).removeprefix("torch.")


def _display_device(value: object) -> str:
    return value.type if isinstance(value, torch.device) else str(value)


def runtime_config_from_environment() -> RuntimeConfig:
    return RuntimeConfig(
        model_id=os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID),
        device=os.getenv("LLM_DEVICE", "auto"),
        dtype=os.getenv("LLM_DTYPE", "auto"),
    )


def create_app(
    runtime: ModelRuntime | None = None,
    *,
    auto_load: bool = True,
) -> FastAPI:
    model_runtime = runtime or ModelRuntime(runtime_config_from_environment())
    metrics = ServingMetrics.create()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        load_task: asyncio.Task[None] | None = None
        if auto_load:
            load_task = asyncio.create_task(model_runtime.load())
        yield
        if load_task is not None and not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task

    application = FastAPI(
        title="Low-Latency LLM Serving API",
        version="0.6.0",
        description="Validated local LLM inference with optional SSE streaming.",
        lifespan=lifespan,
    )
    application.state.runtime = model_runtime
    application.state.metrics = metrics

    @application.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        content, content_type = metrics.render()
        return Response(content=content, media_type=content_type)

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(model_status=model_runtime.status)

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["operations"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Not ready"}},
    )
    async def ready() -> ReadyResponse:
        if model_runtime.status != "ready" or model_runtime.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "model_status": model_runtime.status,
                    "error": model_runtime.error,
                },
            )
        return ReadyResponse(model_id=model_runtime.model.model_id)

    @application.get(
        "/v1/metadata",
        response_model=MetadataResponse,
        tags=["inference"],
    )
    async def metadata() -> MetadataResponse:
        if model_runtime.status != "ready" or model_runtime.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model is not ready",
            )
        model = model_runtime.model
        return MetadataResponse(
            model_id=model.model_id,
            device=_display_device(model.device),
            dtype=_display_dtype(model.dtype),
            load_seconds=model.load_seconds,
        )

    @application.post(
        "/v1/generate",
        response_model=GenerateResponse,
        tags=["inference"],
    )
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        request_started = perf_counter()
        if model_runtime.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model is not ready",
            )

        settings = GenerationSettings(
            max_new_tokens=payload.max_new_tokens,
            do_sample=payload.do_sample,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
        )
        endpoint = "generate"
        metrics.in_progress.labels(endpoint).inc()
        try:
            result = await model_runtime.generate(payload.prompt, settings)
        except RuntimeError as exc:
            metrics.requests.labels(endpoint, "error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        finally:
            metrics.in_progress.labels(endpoint).dec()
            metrics.request_duration.labels(endpoint).observe(
                perf_counter() - request_started
            )

        metrics.requests.labels(endpoint, "success").inc()
        metrics.output_tokens.labels(endpoint).inc(result.output_tokens)
        metrics.batch_size.observe(result.batch_size)

        return GenerateResponse(
            request_id=request.headers.get("x-request-id", str(uuid4())),
            model_id=result.model_id,
            device=result.device,
            dtype=result.dtype,
            response=result.response,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_seconds=result.generation_seconds,
            total_request_seconds=perf_counter() - request_started,
            tokens_per_second=result.tokens_per_second,
            batch_size=result.batch_size,
            parameters=GenerationParameters(**asdict(result.settings)),
        )

    @application.post("/v1/generate/stream", tags=["inference"])
    async def stream_generate(
        payload: GenerateRequest, request: Request
    ) -> StreamingResponse:
        if model_runtime.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model is not ready",
            )
        request_id = request.headers.get("x-request-id", str(uuid4()))
        settings = GenerationSettings(
            max_new_tokens=payload.max_new_tokens,
            do_sample=payload.do_sample,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
        )

        def event(name: str, data: dict[str, object]) -> str:
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        async def events():
            started = perf_counter()
            first_chunk_seconds: float | None = None
            output = ""
            outcome = "success"
            metrics.in_progress.labels("stream").inc()
            yield event("start", {"request_id": request_id})
            try:
                async for chunk in model_runtime.stream_generate(
                    payload.prompt, settings
                ):
                    if first_chunk_seconds is None:
                        first_chunk_seconds = perf_counter() - started
                        metrics.first_chunk.observe(first_chunk_seconds)
                    output += chunk
                    metrics.output_characters.inc(len(chunk))
                    yield event("token", {"text": chunk})
                total_seconds = perf_counter() - started
                yield event(
                    "done",
                    {
                        "request_id": request_id,
                        "response": output.strip(),
                        "time_to_first_chunk_seconds": first_chunk_seconds,
                        "total_request_seconds": total_seconds,
                    },
                )
            except RuntimeError as exc:
                outcome = "error"
                yield event("error", {"request_id": request_id, "detail": str(exc)})
            finally:
                metrics.in_progress.labels("stream").dec()
                metrics.requests.labels("stream", outcome).inc()
                metrics.request_duration.labels("stream").observe(
                    perf_counter() - started
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


app = create_app()
