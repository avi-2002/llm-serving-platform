"""Validated public request and response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=8_000)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    do_sample: bool = False
    temperature: float = Field(default=0.7, gt=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must contain non-whitespace text")
        return value


class GenerationParameters(BaseModel):
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    seed: int


class GenerateResponse(BaseModel):
    request_id: str
    replica_id: str | None = None
    model_id: str
    device: str
    dtype: str
    response: str
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    total_request_seconds: float
    tokens_per_second: float
    parameters: GenerationParameters


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_status: Literal["loading", "ready", "error"]


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_id: str


class MetadataResponse(BaseModel):
    model_id: str
    device: str
    dtype: str
    load_seconds: float
