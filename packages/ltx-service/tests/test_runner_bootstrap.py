from pathlib import Path

import pytest

from ltx_service.config import ModelPaths, RuntimeConfig, expected_model_paths
from ltx_service.runner import DubbingRuntime


def test_runtime_bootstraps_when_flag_is_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    config = RuntimeConfig(
        model_root=model_root,
        scratch_root=tmp_path / "scratch",
        request_connect_timeout_seconds=30,
        request_read_timeout_seconds=30,
        request_chunk_size=1024,
        model_paths=expected_model_paths(model_root),
    )
    ensure_calls = {"count": 0}
    bootstrap_calls: list[Path] = []

    def fake_ensure_model_paths(model_paths: ModelPaths) -> ModelPaths:
        ensure_calls["count"] += 1
        if ensure_calls["count"] == 1:
            raise FileNotFoundError("missing")
        return model_paths

    def fake_bootstrap_models(model_root: Path) -> ModelPaths:
        bootstrap_calls.append(model_root)
        return expected_model_paths(model_root)

    monkeypatch.setenv("LTX_BOOTSTRAP_IF_MISSING", "1")
    monkeypatch.setattr("ltx_service.runner.ensure_model_paths", fake_ensure_model_paths)
    monkeypatch.setattr("ltx_service.runner.bootstrap_models", fake_bootstrap_models)
    monkeypatch.setattr("ltx_service.runner.require_cuda", lambda: "cuda")

    runtime = DubbingRuntime(config)

    assert runtime.device == "cuda"
    assert bootstrap_calls == [model_root.resolve()]
