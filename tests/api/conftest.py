from collections.abc import Iterator

import torch
from fastapi.testclient import TestClient

from low_latency_llm_serving.api.app import create_app
from low_latency_llm_serving.api.runtime import ModelRuntime
from low_latency_llm_serving.inference import (
    GenerationSettings,
    InferenceResult,
)


class FakeLLM:
    model_id = "test/tiny-model"
    device = torch.device("cpu")
    dtype = torch.float32
    load_seconds = 0.125

    def generate(
        self, prompt: str, settings: GenerationSettings
    ) -> InferenceResult:
        return InferenceResult(
            model_id=self.model_id,
            device="cpu",
            dtype="float32",
            prompt=prompt,
            response="A deterministic test response.",
            input_tokens=5,
            output_tokens=6,
            load_seconds=self.load_seconds,
            generation_seconds=0.25,
            tokens_per_second=24.0,
            process_rss_mb=100.0,
            settings=settings,
            batch_size=1,
        )

    def stream_generate(
        self, prompt: str, settings: GenerationSettings
    ) -> Iterator[str]:
        yield "A deterministic "
        yield "test response."


def make_ready_client() -> Iterator[TestClient]:
    runtime = ModelRuntime()
    runtime.install_model(FakeLLM())
    with TestClient(create_app(runtime, auto_load=False)) as client:
        yield client
