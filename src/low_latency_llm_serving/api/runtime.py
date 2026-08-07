"""Lifecycle and synchronization for one local model instance."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

from low_latency_llm_serving.inference import (
    DEFAULT_MODEL_ID,
    GenerationSettings,
    InferenceResult,
    LocalLLM,
)

ModelStatus = Literal["loading", "ready", "error"]


class InferenceModel(Protocol):
    model_id: str
    device: object
    dtype: object
    load_seconds: float

    def generate(
        self, prompt: str, settings: GenerationSettings
    ) -> InferenceResult: ...

    def stream_generate(
        self, prompt: str, settings: GenerationSettings
    ) -> Iterator[str]: ...


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"


class ModelRuntime:
    """Own one model and serialize access to its mutable generation state."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.status: ModelStatus = "loading"
        self.model: InferenceModel | None = None
        self.error: str | None = None
        self.generation_lock = asyncio.Lock()

    async def load(self) -> None:
        self.status = "loading"
        self.error = None
        try:
            self.model = await asyncio.to_thread(
                LocalLLM,
                model_id=self.config.model_id,
                device=self.config.device,
                dtype=self.config.dtype,
            )
            self.status = "ready"
        # Loading crosses filesystem, deserialization, device, and third-party
        # library boundaries. Any failure must become observable readiness state.
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"

    def install_model(self, model: InferenceModel) -> None:
        """Install an already-created model, primarily for fast API tests."""
        self.model = model
        self.status = "ready"
        self.error = None

    async def generate(
        self, prompt: str, settings: GenerationSettings
    ) -> InferenceResult:
        if self.status != "ready" or self.model is None:
            raise RuntimeError("model is not ready")

        async with self.generation_lock:
            return await asyncio.to_thread(self.model.generate, prompt, settings)

    async def stream_generate(
        self, prompt: str, settings: GenerationSettings
    ) -> AsyncIterator[str]:
        if self.status != "ready" or self.model is None:
            raise RuntimeError("model is not ready")

        sentinel = object()

        def next_chunk(iterator: Iterator[str]) -> str | object:
            return next(iterator, sentinel)

        async with self.generation_lock:
            iterator = self.model.stream_generate(prompt, settings)
            while True:
                chunk = await asyncio.to_thread(next_chunk, iterator)
                if chunk is sentinel:
                    break
                yield str(chunk)
