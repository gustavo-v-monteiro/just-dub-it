from pathlib import Path

from ltx_service.bootstrap import verify_model_paths
from ltx_service.config import expected_model_paths


def test_verify_model_paths_reports_missing_files(tmp_path: Path) -> None:
    model_paths = expected_model_paths(tmp_path)
    missing = verify_model_paths(model_paths)

    assert "checkpoint_path" in missing
    assert "gemma_root" in missing


def test_verify_model_paths_accepts_expected_layout(tmp_path: Path) -> None:
    model_paths = expected_model_paths(tmp_path)
    for path in (
        model_paths.checkpoint_path,
        model_paths.justdubit_lora_path,
        model_paths.distilled_lora_path,
        model_paths.spatial_upsampler_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    model_paths.gemma_root.mkdir(parents=True, exist_ok=True)
    (model_paths.gemma_root / "config.json").write_text("{}")

    assert verify_model_paths(model_paths) == []
