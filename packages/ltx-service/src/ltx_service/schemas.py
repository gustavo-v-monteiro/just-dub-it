from typing import Any

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

from ltx_pipelines.constants import (
    DEFAULT_CFG_GUIDANCE_SCALE,
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
)


class DubbingJobRequest(BaseModel):
    """Portable request schema shared by provider adapters."""

    model_config = ConfigDict(extra="forbid")

    input_video_url: AnyHttpUrl
    output_video_url: AnyHttpUrl
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    frame_rate: float = DEFAULT_FRAME_RATE
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    cfg_guidance_scale: float = DEFAULT_CFG_GUIDANCE_SCALE
    seed: int = DEFAULT_SEED
    conditioning_strength: float = Field(default=1.0)
    enable_fp8: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value

    @field_validator("height", "width")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dimensions must be positive")
        if value % 32 != 0:
            raise ValueError("height and width must be multiples of 32")
        return value

    @field_validator("frame_rate")
    @classmethod
    def validate_frame_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("frame_rate must be positive")
        return value

    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_inference_steps must be positive")
        return value

    @field_validator("cfg_guidance_scale")
    @classmethod
    def validate_cfg_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cfg_guidance_scale must be positive")
        return value

    @field_validator("conditioning_strength")
    @classmethod
    def validate_conditioning_strength(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("conditioning_strength must be between 0.0 and 1.0")
        return value


class DubbingJobResult(BaseModel):
    """Normalized response payload for completed dubbing jobs."""

    model_config = ConfigDict(extra="forbid")

    output_video_url: AnyHttpUrl
    seed: int
    input_frames: int
    stage1_width: int
    stage1_height: int
    final_width: int
    final_height: int
    frame_rate: float
    duration_seconds: float
    provider_job_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None

    @classmethod
    def failed(
        cls,
        *,
        request: DubbingJobRequest,
        provider_job_id: str | None,
        error: Exception,
        input_frames: int = 0,
    ) -> "DubbingJobResult":
        return cls(
            output_video_url=request.output_video_url,
            seed=request.seed,
            input_frames=input_frames,
            stage1_width=request.width,
            stage1_height=request.height,
            final_width=request.width * 2,
            final_height=request.height * 2,
            frame_rate=request.frame_rate,
            duration_seconds=0.0,
            provider_job_id=provider_job_id,
            error_code=type(error).__name__,
            error_message=str(error),
            error_details=None,
        )
