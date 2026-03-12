"""
FastAPI wrapper for JustDubit.

Wraps JustDubitPipeline in a REST API that accepts a source video file upload
plus a text prompt and returns a dubbed MP4 video.

The pipeline is loaded once at startup and kept warm in memory between requests.
All model paths are read from environment variables (set by modal_deploy.py).

Environment variables required:
    CHECKPOINT_PATH          – path to ltx-2-19b-dev.safetensors
    GEMMA_ROOT               – path to the Gemma text-encoder directory
    DISTILLED_LORA_PATH      – path to ltx-2-19b-distilled-lora-384.safetensors
    SPATIAL_UPSAMPLER_PATH   – path to ltx-2-spatial-upscaler-x2-1.0.safetensors
    JUSTDUBIT_LORA_PATH      – path to ltx-2-19b-ic-lora-lipdubbing.safetensors
    JUSTDUBIT_API_KEY        – secret key clients must supply via X-API-Key header

Usage (local):
    uvicorn serve:app --host 0.0.0.0 --port 8000

Usage (Modal):
    Imported dynamically by modal_deploy.py – see that file for details.
"""

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig
from ltx_pipelines.pipeline_justdubit import JustDubitPipeline
from ltx_pipelines.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.media_io import encode_video

# ── Model paths ───────────────────────────────────────────────────────────────
# All paths come from environment variables set by the Modal deployment script.
CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
GEMMA_ROOT = os.environ["GEMMA_ROOT"]
DISTILLED_LORA_PATH = os.environ["DISTILLED_LORA_PATH"]
SPATIAL_UPSAMPLER_PATH = os.environ["SPATIAL_UPSAMPLER_PATH"]
JUSTDUBIT_LORA_PATH = os.environ["JUSTDUBIT_LORA_PATH"]

# ── API key auth ──────────────────────────────────────────────────────────────
# Clients must pass the key in the X-API-Key request header.
# Set JUSTDUBIT_API_KEY as a Modal secret named "justdubit-api-key".
API_KEY = os.environ.get("JUSTDUBIT_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(key: str = Security(api_key_header)) -> None:
    """Raise HTTP 403 if the provided key does not match JUSTDUBIT_API_KEY."""
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Pipeline (loaded once at startup via lifespan) ────────────────────────────
pipeline: JustDubitPipeline | None = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):  # noqa: ANN201
    """Load the JustDubit pipeline into GPU memory once at startup."""
    global pipeline
    pipeline = JustDubitPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora_path=DISTILLED_LORA_PATH,
        distilled_lora_strength=1.0,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[
            LoraPathStrengthAndSDOps(JUSTDUBIT_LORA_PATH, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),
        ],
    )
    yield
    # Nothing to clean up on shutdown (GPU memory is freed when the process exits).


app = FastAPI(title="JustDubit API", lifespan=lifespan)

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe – returns {"status": "ok"}."""
    return {"status": "ok"}


@app.post("/dub", dependencies=[Security(verify_api_key)])
async def dub_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Source MP4 video to dub"),
    prompt: str = Form(..., description="Text prompt describing the dubbed speech"),
    negative_prompt: str = Form(
        "blurry, low quality, distorted",
        description="Negative prompt passed to the diffusion model",
    ),
    height: int = Form(512, description="Output frame height in pixels"),
    width: int = Form(768, description="Output frame width in pixels"),
    num_frames: int = Form(121, description="Number of video frames to generate"),
    frame_rate: float = Form(25.0, description="Output frames per second"),
    num_inference_steps: int = Form(30, description="Diffusion denoising steps"),
    cfg_guidance_scale: float = Form(3.0, description="Classifier-free guidance scale"),
    seed: int = Form(42, description="Random seed for reproducibility"),
) -> FileResponse:
    """Upload a source video and a text prompt; receive a dubbed MP4 in return.

    Example curl call:
        curl -X POST "https://<your-modal-url>/dub" \\
          -H "X-API-Key: your-secret-key" \\
          -F "video=@source_video.mp4" \\
          -F "prompt=The man is speaking French, saying: 'Bonjour le monde!'" \\
          --output dubbed_output.mp4
    """
    # Write the uploaded video to a temp file so the pipeline can read it.
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / f"input_{uuid.uuid4().hex}.mp4"
        out_path = Path(tmp) / f"output_{uuid.uuid4().hex}.mp4"

        # Stream the uploaded video to disk in chunks to avoid high memory usage.
        with src_path.open("wb") as dst:
            while True:
                chunk = await video.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                dst.write(chunk)

        # Run the two-stage JustDubit inference pipeline.
        global pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Model pipeline is not available yet; please retry later.",
            )
        video_out, audio_out = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
            images=[],
            video_conditioning=[(str(src_path), 1.0)],
            tiling_config=TilingConfig.default(),
        )

        # Encode the output tensors to an MP4 file.
        encode_video(
            video=video_out,
            fps=int(frame_rate),
            audio=audio_out,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=str(out_path),
        )

        # Read the file into memory so it survives the TemporaryDirectory cleanup.
        video_bytes = out_path.read_bytes()

    # Write to a second temp file outside the deleted directory and stream it.
    final_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    final_tmp.write(video_bytes)
    final_tmp.flush()

    # Schedule deletion of the temp file after the response is fully sent.
    background_tasks.add_task(os.unlink, final_tmp.name)

    return FileResponse(
        final_tmp.name,
        media_type="video/mp4",
        filename="dubbed.mp4",
    )
