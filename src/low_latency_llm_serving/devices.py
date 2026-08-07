"""Device and numeric-precision selection.

Keeping this policy outside the model class makes the hardware decision easy to
test and, later, easy to replace when Ray Serve assigns resources to replicas.
"""

from __future__ import annotations

import torch

SUPPORTED_DEVICES = ("auto", "cpu", "mps")
SUPPORTED_DTYPES = ("auto", "float32", "float16")


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {requested!r}; choose one of {SUPPORTED_DEVICES}."
        )

    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable in this Python process. "
            "Use --device cpu or run from a Terminal where MPS is available."
        )

    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Choose a conservative default precision for the selected device."""
    if requested not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {requested!r}; choose one of {SUPPORTED_DTYPES}."
        )

    if requested == "auto":
        return torch.float16 if device.type == "mps" else torch.float32

    return {"float32": torch.float32, "float16": torch.float16}[requested]


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()

