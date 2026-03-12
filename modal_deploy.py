"""
Modal deployment configuration for JustDubit.

Defines the container image, persistent model volume, a one-time model-download
helper, and the GPU-backed FastAPI web endpoint.

Deployment steps (run from any machine with Modal installed):

    pip install modal
    modal token new                                  # authenticate once via browser
    modal run modal_deploy.py::download_models       # one-time: download ~40 GB of model weights
    modal deploy modal_deploy.py                     # deploy (or re-deploy) the GPU endpoint

After deployment Modal prints a URL like:
    https://<workspace>--justdubit-fastapi-app.modal.run

Call the endpoint:
    curl -X POST "https://<your-modal-url>/dub" \\
      -H "X-API-Key: your-secret-key" \\
      -F "video=@source_video.mp4" \\
      -F "prompt=The man is speaking French, saying: 'Bonjour le monde!'" \\
      --output dubbed_output.mp4
"""

import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

# ── 1. Container image ────────────────────────────────────────────────────────
# Build a Debian-slim image with all system and Python dependencies baked in.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # System libraries required by ffmpeg / PyAV and the git clone below.
    .apt_install("git", "ffmpeg", "libavcodec-dev", "libavformat-dev")
    # Install the uv package manager used by this workspace.
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh && ln -s /root/.local/bin/uv /usr/local/bin/uv",
    )
    # Clone the repo and install all workspace packages (ltx-core, ltx-pipelines) into the system interpreter.
    .run_commands(
        "git clone https://github.com/gustavo-v-monteiro/just-dub-it.git /app && cd /app && uv sync --frozen --system",
    )
    # Install the API-serving extras on top of the uv-managed environment.
    .pip_install("fastapi", "uvicorn[standard]", "python-multipart", "huggingface_hub")
)

# ── 2. Persistent volume for model weights ────────────────────────────────────
# All model checkpoints (~40 GB total) are stored here once and reused across
# every container invocation, avoiding repeated large downloads.
volume = modal.Volume.from_name("justdubit-models", create_if_missing=True)
MODEL_DIR = "/models"

# ── 3. Modal app ──────────────────────────────────────────────────────────────
app = modal.App("justdubit", image=image)


# ── 4. One-time model download ────────────────────────────────────────────────
@app.function(volumes={MODEL_DIR: volume}, timeout=3600)
def download_models() -> None:
    """Download all required model weights into the persistent Modal Volume.

    Run this once before deploying the endpoint:
        modal run modal_deploy.py::download_models

    Downloads (~40 GB total):
        - LTX-2 main checkpoint                (Lightricks/LTX-2)
        - LTX-2 distilled LoRA                 (Lightricks/LTX-2)
        - LTX-2 spatial upscaler               (Lightricks/LTX-2)
        - JustDubit lip-dubbing LoRA           (justdubit/justdubit)
        - Gemma 3 12B text encoder (full dir)  (google/gemma-3-12b-it-qat-q4_0-unquantized)
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Downloading LTX-2 main checkpoint…")
    hf_hub_download(
        repo_id="Lightricks/LTX-2",
        filename="ltx-2-19b-dev.safetensors",
        local_dir=MODEL_DIR,
    )

    print("Downloading LTX-2 distilled LoRA…")
    hf_hub_download(
        repo_id="Lightricks/LTX-2",
        filename="ltx-2-19b-distilled-lora-384.safetensors",
        local_dir=MODEL_DIR,
    )

    print("Downloading LTX-2 spatial upscaler…")
    hf_hub_download(
        repo_id="Lightricks/LTX-2",
        filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        local_dir=MODEL_DIR,
    )

    print("Downloading JustDubit lip-dubbing LoRA…")
    hf_hub_download(
        repo_id="justdubit/justdubit",
        filename="ltx-2-19b-ic-lora-lipdubbing.safetensors",
        local_dir=MODEL_DIR,
    )

    # The Gemma model is a full directory snapshot (~8 GB).
    # If it is a gated model you may need to add a HuggingFace token secret:
    #   secrets=[modal.Secret.from_name("huggingface-token")]
    # to this function's @app.function decorator and set HF_TOKEN in that secret.
    print("Downloading Gemma 3 12B text encoder (this may take a while)…")
    snapshot_download(
        repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
        local_dir=f"{MODEL_DIR}/gemma",
    )

    # Persist all downloaded files to the volume so other containers see them.
    volume.commit()
    print("✅ All models downloaded and committed to the Modal Volume!")


# ── 5. FastAPI web endpoint ───────────────────────────────────────────────────
# Note: if the Gemma model requires a HuggingFace token (gated model), create a
# Modal secret named "huggingface-token" containing HF_TOKEN and add it here:
#   secrets=[modal.Secret.from_name("huggingface-token")]
#
# To protect the endpoint with an API key, create a Modal secret named
# "justdubit-api-key" containing JUSTDUBIT_API_KEY and add it here:
#   secrets=[modal.Secret.from_name("justdubit-api-key")]
@app.function(
    gpu=modal.gpu.A100(size="40GB"),  # change to "80GB" or modal.gpu.H100() for larger models
    volumes={MODEL_DIR: volume},
    timeout=600,  # 10-minute hard limit per request
    container_idle_timeout=300,  # keep container warm for 5 minutes after last request
)
@modal.concurrent(max_inputs=1)  # one inference job at a time per container (GPU-bound)
@modal.asgi_app()
def fastapi_app() -> "FastAPI":  # return type annotation for clarity
    """Return the JustDubit FastAPI application for Modal's ASGI gateway.

    Environment variables pointing to model paths under the persistent volume
    are set here before serve.py is imported so that module-level code in
    serve.py can read them at import time.
    """
    # Expose model paths to serve.py via environment variables.
    os.environ["CHECKPOINT_PATH"] = f"{MODEL_DIR}/ltx-2-19b-dev.safetensors"
    os.environ["GEMMA_ROOT"] = f"{MODEL_DIR}/gemma"
    os.environ["DISTILLED_LORA_PATH"] = f"{MODEL_DIR}/ltx-2-19b-distilled-lora-384.safetensors"
    os.environ["SPATIAL_UPSAMPLER_PATH"] = f"{MODEL_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    os.environ["JUSTDUBIT_LORA_PATH"] = f"{MODEL_DIR}/ltx-2-19b-ic-lora-lipdubbing.safetensors"

    # Make the cloned repo's source tree importable.
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Dynamically import serve.py from the cloned repo so that it picks up the
    # environment variables we just set before executing its module-level code.
    spec = importlib.util.spec_from_file_location("serve", "/app/serve.py")
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load module 'serve' from '/app/serve.py'. "
            "Ensure the repository was cloned correctly and the file exists."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Return the FastAPI app object for Modal's ASGI gateway.
    return module.app
