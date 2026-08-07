import pytest
import torch

from low_latency_llm_serving.devices import resolve_device, resolve_dtype


def test_explicit_cpu_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_dtype_is_float32_on_cpu() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32


def test_explicit_float16_dtype() -> None:
    assert resolve_dtype("float16", torch.device("cpu")) == torch.float16


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("gpu")

