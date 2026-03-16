from pathlib import Path

import pytest
from huggingface_hub.errors import GatedRepoError

from ltx_service.bootstrap import bootstrap_models, verify_model_paths
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


def test_bootstrap_models_falls_back_to_public_gemma_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloads: dict[str, bytes] = {
        "ltx-2-19b-dev.safetensors": b"checkpoint",
        "ltx-2-19b-ic-lora-lipdubbing.safetensors": b"justdubit-lora",
        "ltx-2-19b-distilled-lora-384.safetensors": b"distilled-lora",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors": b"spatial-upscaler",
        "config.json": b"{}",
        "tokenizer.model": b"tokenizer",
        "model.safetensors.index.json": b"{}",
        "gemma-3-12b-it-qat-q4_0-unquantized.safetensors": b"weights",
    }

    def fake_hf_hub_download(
        *, filename: str, local_dir: str | None = None, subfolder: str | None = None, **_: object
    ) -> str:
        path = Path(local_dir or tmp_path / "downloads")
        if subfolder:
            path = path / subfolder
        path = path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(downloads[filename])
        return str(path)

    class FakeApi:
        def __init__(self, token: str | None = None):
            self.token = token

        def list_repo_files(self, repo_id: str) -> list[str]:
            assert repo_id == "DeepBeepMeep/LTX-2"
            return [
                "gemma-3-12b-it-qat-q4_0-unquantized/README.md",
                "gemma-3-12b-it-qat-q4_0-unquantized/config.json",
                "gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.model",
                "gemma-3-12b-it-qat-q4_0-unquantized/model.safetensors.index.json",
                "gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized.safetensors",
            ]

    def fake_snapshot_download(**_: object) -> str:
        raise GatedRepoError("gated")

    monkeypatch.setattr("ltx_service.bootstrap.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("ltx_service.bootstrap.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("ltx_service.bootstrap.HfApi", FakeApi)

    model_paths = bootstrap_models(tmp_path)

    assert model_paths.gemma_root.joinpath("config.json").read_text() == "{}"
    assert model_paths.gemma_root.joinpath("tokenizer.model").read_text() == "tokenizer"
    assert model_paths.gemma_root.joinpath("model.safetensors.index.json").read_text() == "{}"
    assert model_paths.gemma_root.joinpath(
        "gemma-3-12b-it-qat-q4_0-unquantized.safetensors"
    ).read_text() == "weights"
