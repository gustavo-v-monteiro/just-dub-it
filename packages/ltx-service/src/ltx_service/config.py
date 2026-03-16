import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_service.model_manifest import (
    CHECKPOINT_ARTIFACT,
    DISTILLED_LORA_ARTIFACT,
    GEMMA_ARTIFACT,
    JUSTDUBIT_LORA_ARTIFACT,
    SPATIAL_UPSAMPLER_ARTIFACT,
)


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    checkpoint_path: Path
    justdubit_lora_path: Path
    distilled_lora_path: Path
    spatial_upsampler_path: Path
    gemma_root: Path


def expected_model_paths(model_root: Path) -> ModelPaths:
    return ModelPaths(
        root=model_root,
        checkpoint_path=model_root / CHECKPOINT_ARTIFACT.relative_path,
        justdubit_lora_path=model_root / JUSTDUBIT_LORA_ARTIFACT.relative_path,
        distilled_lora_path=model_root / DISTILLED_LORA_ARTIFACT.relative_path,
        spatial_upsampler_path=model_root / SPATIAL_UPSAMPLER_ARTIFACT.relative_path,
        gemma_root=model_root / GEMMA_ARTIFACT.relative_path,
    )


@dataclass(frozen=True)
class RuntimeConfig:
    model_root: Path
    scratch_root: Path
    request_connect_timeout_seconds: int
    request_read_timeout_seconds: int
    request_chunk_size: int
    model_paths: ModelPaths

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        model_root = Path(os.getenv("LTX_MODEL_ROOT", "/models")).expanduser().resolve()
        scratch_root = Path(os.getenv("LTX_SCRATCH_ROOT", "/tmp/jobs")).expanduser().resolve()
        connect_timeout = int(os.getenv("LTX_REQUEST_CONNECT_TIMEOUT_SECONDS", "30"))
        read_timeout = int(os.getenv("LTX_REQUEST_READ_TIMEOUT_SECONDS", "3600"))
        chunk_size = int(os.getenv("LTX_REQUEST_CHUNK_SIZE", str(1024 * 1024)))
        return cls(
            model_root=model_root,
            scratch_root=scratch_root,
            request_connect_timeout_seconds=connect_timeout,
            request_read_timeout_seconds=read_timeout,
            request_chunk_size=chunk_size,
            model_paths=expected_model_paths(model_root),
        )

    @property
    def request_timeout(self) -> tuple[int, int]:
        return (
            self.request_connect_timeout_seconds,
            self.request_read_timeout_seconds,
        )


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for serverless inference but no GPU was detected")
    return torch.device("cuda")


def current_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"
