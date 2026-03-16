import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ltx_service.config import ModelPaths, expected_model_paths
from ltx_service.model_manifest import FILE_ARTIFACTS, GEMMA_ARTIFACT


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
        downloaded_path = Path(
            hf_hub_download(
                repo_id=artifact.repo_id,
                filename=artifact.filename,
                token=token,
                force_download=force,
            )
        )
        shutil.copy2(downloaded_path, destination)

    gemma_destination = model_root / GEMMA_ARTIFACT.relative_path
    if force and gemma_destination.exists():
        shutil.rmtree(gemma_destination)
    if not gemma_destination.exists():
        gemma_destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=GEMMA_ARTIFACT.repo_id,
            local_dir=str(gemma_destination),
            token=token,
            force_download=force,
        )

    model_paths = expected_model_paths(model_root)
    ensure_model_paths(model_paths)
    return model_paths
