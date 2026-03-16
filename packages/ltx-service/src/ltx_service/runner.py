import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import av
import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig
from ltx_pipelines.constants import AUDIO_SAMPLE_RATE, DEFAULT_LORA_STRENGTH
from ltx_pipelines.media_io import encode_video
from ltx_pipelines.pipeline_justdubit import JustDubitPipeline, extract_first_frame
from ltx_service.bootstrap import bootstrap_models, ensure_model_paths
from ltx_service.config import RuntimeConfig, current_gpu_name, require_cuda
from ltx_service.schemas import DubbingJobRequest, DubbingJobResult
from ltx_service.storage import download_to_path, upload_file

_RUNTIME_STATE: dict[str, "DubbingRuntime | None"] = {"runtime": None}
_RUNTIME_LOCK = threading.Lock()


def _log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


def _effective_frame_count(video_path: Path) -> int:
    frame_count = 0
    with av.open(str(video_path)) as container:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        frame_count = int(video_stream.frames or 0)
        if frame_count == 0:
            frame_count = sum(1 for _ in container.decode(video_stream))

    if frame_count <= 0:
        raise ValueError(f"Could not determine frame count for {video_path}")

    if (frame_count - 1) % 8 != 0:
        frame_count = ((frame_count - 1) // 8) * 8 + 1
    return frame_count


class DubbingRuntime:
    """Warm-worker runtime that lazily constructs the pipeline once per container."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        if os.getenv("LTX_BOOTSTRAP_IF_MISSING", "").lower() in {"1", "true", "yes"}:
            try:
                ensure_model_paths(config.model_paths)
            except FileNotFoundError:
                _log_event("runtime.bootstrap.start", model_root=config.model_root)
                bootstrap_models(config.model_root)
                _log_event("runtime.bootstrap.complete", model_root=config.model_root)
        self.device = require_cuda()
        self.model_paths = ensure_model_paths(config.model_paths)
        self._pipeline: JustDubitPipeline | None = None
        self._pipeline_lock = threading.Lock()

    def get_pipeline(self) -> JustDubitPipeline:
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    _log_event(
                        "runtime.pipeline_init.start",
                        checkpoint_path=self.model_paths.checkpoint_path,
                        gpu=current_gpu_name(),
                    )
                    self._pipeline = JustDubitPipeline(
                        checkpoint_path=str(self.model_paths.checkpoint_path),
                        distilled_lora_path=str(self.model_paths.distilled_lora_path),
                        distilled_lora_strength=DEFAULT_LORA_STRENGTH,
                        spatial_upsampler_path=str(self.model_paths.spatial_upsampler_path),
                        gemma_root=str(self.model_paths.gemma_root),
                        loras=[
                            LoraPathStrengthAndSDOps(
                                str(self.model_paths.justdubit_lora_path),
                                DEFAULT_LORA_STRENGTH,
                                LTXV_LORA_COMFY_RENAMING_MAP,
                            )
                        ],
                        device=self.device,
                    )
                    _log_event("runtime.pipeline_init.complete", gpu=current_gpu_name())
        return self._pipeline

    def run(self, request: DubbingJobRequest, *, provider_job_id: str | None = None) -> DubbingJobResult:
        pipeline = self.get_pipeline()
        job_id = provider_job_id or f"job-{uuid.uuid4().hex}"
        job_root = self.config.scratch_root / job_id
        input_path = job_root / "input.mp4"
        first_frame_path = job_root / "first_frame.png"
        output_path = job_root / "output.mp4"
        started_at = time.perf_counter()

        _log_event(
            "job.start",
            job_id=job_id,
            input_video_url=request.input_video_url,
            output_video_url=request.output_video_url,
            height=request.height,
            width=request.width,
            frame_rate=request.frame_rate,
            num_inference_steps=request.num_inference_steps,
            gpu=current_gpu_name(),
        )

        job_root.mkdir(parents=True, exist_ok=True)

        try:
            download_to_path(
                str(request.input_video_url),
                input_path,
                timeout=self.config.request_timeout,
                chunk_size=self.config.request_chunk_size,
            )
            input_frames = _effective_frame_count(input_path)
            extract_first_frame(str(input_path), str(first_frame_path))

            _log_event("job.inference.start", job_id=job_id, input_frames=input_frames)
            video, audio = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=input_frames,
                frame_rate=request.frame_rate,
                num_inference_steps=request.num_inference_steps,
                cfg_guidance_scale=request.cfg_guidance_scale,
                images=[(str(first_frame_path), 0, request.conditioning_strength)],
                video_conditioning=[(str(input_path), request.conditioning_strength)],
                tiling_config=TilingConfig.default(),
            )
            _log_event("job.inference.complete", job_id=job_id)

            encode_video(
                video=video,
                fps=int(request.frame_rate),
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_path),
            )
            upload_file(
                str(request.output_video_url),
                output_path,
                timeout=self.config.request_timeout,
            )

            elapsed_seconds = time.perf_counter() - started_at
            result = DubbingJobResult(
                output_video_url=request.output_video_url,
                seed=request.seed,
                input_frames=input_frames,
                stage1_width=request.width,
                stage1_height=request.height,
                final_width=request.width * 2,
                final_height=request.height * 2,
                frame_rate=request.frame_rate,
                duration_seconds=elapsed_seconds,
                provider_job_id=provider_job_id,
            )
            _log_event(
                "job.complete",
                job_id=job_id,
                duration_seconds=elapsed_seconds,
                input_frames=input_frames,
                final_width=result.final_width,
                final_height=result.final_height,
            )
            return result
        finally:
            shutil.rmtree(job_root, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_runtime(config: RuntimeConfig | None = None) -> DubbingRuntime:
    runtime_config = config or RuntimeConfig.from_env()
    if _RUNTIME_STATE["runtime"] is None:
        with _RUNTIME_LOCK:
            if _RUNTIME_STATE["runtime"] is None:
                _RUNTIME_STATE["runtime"] = DubbingRuntime(runtime_config)
    runtime = _RUNTIME_STATE["runtime"]
    if runtime is None:
        raise RuntimeError("runtime initialization failed")
    return runtime


def run_job(request: DubbingJobRequest, *, provider_job_id: str | None = None) -> DubbingJobResult:
    return get_runtime().run(request, provider_job_id=provider_job_id)
