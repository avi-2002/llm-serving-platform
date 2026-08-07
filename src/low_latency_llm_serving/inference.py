"""Local causal-language-model inference and baseline measurements."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from threading import Thread
from time import perf_counter

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from low_latency_llm_serving.devices import resolve_device, resolve_dtype, synchronize

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class GenerationSettings:
    """Parameters that alter autoregressive decoding."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")


@dataclass(frozen=True)
class InferenceResult:
    """Response plus the minimum useful Phase 1 performance measurements."""

    model_id: str
    device: str
    dtype: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    load_seconds: float
    generation_seconds: float
    tokens_per_second: float
    process_rss_mb: float
    settings: GenerationSettings
    batch_size: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LocalLLM:
    """Load one model once, then generate multiple responses with it."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)

        load_started = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        synchronize(self.device)
        self.load_seconds = perf_counter() - load_started

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
    ) -> InferenceResult:
        settings = settings or GenerationSettings()
        settings.validate()
        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")

        torch.manual_seed(settings.seed)
        messages = [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = model_inputs["input_ids"]

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.do_sample:
            generation_kwargs.update(
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        else:
            # Qwen's stored generation config contains sampling defaults. Explicitly
            # clearing them prevents misleading warnings during greedy decoding.
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)

        synchronize(self.device)
        generation_started = perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)
        synchronize(self.device)
        generation_seconds = perf_counter() - generation_started

        input_tokens = input_ids.shape[-1]
        generated_ids = output_ids[0, input_tokens:]
        output_tokens = generated_ids.shape[-1]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_per_second = (
            output_tokens / generation_seconds if generation_seconds else 0.0
        )
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        return InferenceResult(
            model_id=self.model_id,
            device=self.device.type,
            dtype=str(self.dtype).removeprefix("torch."),
            prompt=prompt,
            response=response.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            load_seconds=self.load_seconds,
            generation_seconds=generation_seconds,
            tokens_per_second=tokens_per_second,
            process_rss_mb=rss_mb,
            settings=settings,
        )

    def stream_generate(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
    ) -> Iterator[str]:
        """Yield decoded text as generation progresses."""
        settings = settings or GenerationSettings()
        settings.validate()
        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")

        torch.manual_seed(settings.seed)
        messages = [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0,
        )
        generation_kwargs: dict[str, object] = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if settings.do_sample:
            generation_kwargs.update(
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)

        generation_error: list[Exception] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
                synchronize(self.device)
            # Generation crosses model, device, and streamer boundaries. Any
            # failure must unblock the consumer and then be propagated.
            except Exception as exc:  # noqa: BLE001
                generation_error.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        for text in streamer:
            if text:
                yield text
        worker.join()
        if generation_error:
            raise RuntimeError("streaming generation failed") from generation_error[0]

    def generate_batch(
        self,
        prompts: list[str],
        settings: GenerationSettings | None = None,
    ) -> list[InferenceResult]:
        """Generate one response per prompt in a single tensor batch."""
        settings = settings or GenerationSettings()
        settings.validate()
        if not prompts:
            raise ValueError("prompts must contain at least one item")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("every prompt must contain non-whitespace text")

        torch.manual_seed(settings.seed)
        conversations = [
            [
                {
                    "role": "system",
                    "content": "You are a concise, helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            model_inputs = self.tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = original_padding_side

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if settings.do_sample:
            generation_kwargs.update(
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)

        synchronize(self.device)
        generation_started = perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)
        synchronize(self.device)
        generation_seconds = perf_counter() - generation_started

        padded_input_length = model_inputs["input_ids"].shape[-1]
        generated_batch = output_ids[:, padded_input_length:]
        input_token_counts = model_inputs["attention_mask"].sum(dim=1).tolist()
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        batch_size = len(prompts)
        results: list[InferenceResult] = []

        for prompt, generated_ids, input_tokens in zip(
            prompts, generated_batch, input_token_counts, strict=True
        ):
            non_padding = generated_ids != self.tokenizer.pad_token_id
            valid_ids = generated_ids[non_padding]
            output_tokens = int(non_padding.sum().item())
            response = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            results.append(
                InferenceResult(
                    model_id=self.model_id,
                    device=self.device.type,
                    dtype=str(self.dtype).removeprefix("torch."),
                    prompt=prompt,
                    response=response.strip(),
                    input_tokens=int(input_tokens),
                    output_tokens=output_tokens,
                    load_seconds=self.load_seconds,
                    generation_seconds=generation_seconds,
                    tokens_per_second=(
                        output_tokens / generation_seconds
                        if generation_seconds
                        else 0.0
                    ),
                    process_rss_mb=rss_mb,
                    settings=settings,
                    batch_size=batch_size,
                )
            )

        return results
