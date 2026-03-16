import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

from ltx_service.config import ModelPaths, expected_model_paths
from ltx_service.model_manifest import (
    FILE_ARTIFACTS,
    GEMMA_ARTIFACT,
    GEMMA_PUBLIC_MIRROR_PREFIX,
    GEMMA_PUBLIC_MIRROR_REPO_ID,
    GEMMA_PUBLIC_MIRROR_REQUIRED_FILES,
)


def verify_model_paths(model_paths: ModelPaths) -> list[str]:
    missing: list[str] = []

    for name, path in (
        ("checkpoint_path", model_paths.checkpoint_path),
        ("justdubit_lora_path", model_paths.justdubit_lora_path),
        ("distilled_lora_path", model_paths.distilled_lora_path),
        ("spatial_upsampler_path", model_paths.spatial_upsampler_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(name)

    if not model_paths.gemma_root.is_dir():
        missing.append("gemma_root")
    else:
        for marker in GEMMA_ARTIFACT.required_markers:
            if not (model_paths.gemma_root / marker).exists():
                missing.append(f"gemma_root:{marker}")

    return missing


def ensure_model_paths(model_paths: ModelPaths) -> ModelPaths:
    missing = verify_model_paths(model_paths)
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts: "
            + ", ".join(missing)
            + ". Run `ltx-service bootstrap` first."
        )
    return model_paths


def _download_gemma_public_mirror(destination: Path, *, token: str | None = None, force: bool = False) -> None:
    api = HfApi(token=token)
    prefix = f"{GEMMA_PUBLIC_MIRROR_PREFIX}/"
    required_paths = {f"{prefix}{filename}" for filename in GEMMA_PUBLIC_MIRROR_REQUIRED_FILES}
    files = [path for path in api.list_repo_files(GEMMA_PUBLIC_MIRROR_REPO_ID) if path in required_paths]
    if not files:
        raise FileNotFoundError(f"No files found under {GEMMA_PUBLIC_MIRROR_REPO_ID}:{prefix}")

    for repo_path in files:
        relative_path = Path(repo_path.removeprefix(prefix))
        hf_hub_download(
            repo_id=GEMMA_PUBLIC_MIRROR_REPO_ID,
            filename=relative_path.name,
            subfolder=GEMMA_PUBLIC_MIRROR_PREFIX,
            token=token,
            force_download=force,
            local_dir=str(destination.parent),
        )


def bootstrap_models(
    model_root: Path,
    *,
    token: str | None = None,
    force: bool = False,
) -> ModelPaths:
    model_root = model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    for artifact in FILE_ARTIFACTS:
        destination = model_root / artifact.relative_path
        if destination.exists() and not force:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=artifact.repo_id,
            filename=artifact.filename,
            token=token,
            force_download=force,
            local_dir=str(destination.parent),
        )

    gemma_destination = model_root / GEMMA_ARTIFACT.relative_path
    if force and gemma_destination.exists():
        shutil.rmtree(gemma_destination)
    if not gemma_destination.exists():
        gemma_destination.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=GEMMA_ARTIFACT.repo_id,
                local_dir=str(gemma_destination),
                token=token,
                force_download=force,
            )
        except GatedRepoError:
            _download_gemma_public_mirror(gemma_destination, token=token, force=force)

    model_paths = expected_model_paths(model_root)
    ensure_model_paths(model_paths)
    return model_paths
