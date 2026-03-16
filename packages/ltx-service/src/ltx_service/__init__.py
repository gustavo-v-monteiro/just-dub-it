"""Portable serverless runtime for JustDubit inference."""

import importlib

from ltx_service.config import ModelPaths, RuntimeConfig
from ltx_service.schemas import DubbingJobRequest, DubbingJobResult

__all__ = [
    "DubbingJobRequest",
    "DubbingJobResult",
    "DubbingRuntime",
    "ModelPaths",
    "RuntimeConfig",
    "get_runtime",
    "run_job",
]


def __getattr__(name: str) -> object:
    if name in {"DubbingRuntime", "get_runtime", "run_job"}:
        module = importlib.import_module("ltx_service.runner")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
