import pytest

from ltx_service.providers.runpod_handler import handler, parse_runpod_event
from ltx_service.schemas import DubbingJobResult


def test_parse_runpod_event_extracts_input_and_job_id() -> None:
    request, provider_job_id = parse_runpod_event(
        {
            "id": "job-123",
            "input": {
                "input_video_url": "https://example.com/input.mp4",
                "output_video_url": "https://example.com/output.mp4",
                "prompt": "The man is speaking English, saying: 'Hello!'",
            },
        }
    )

    assert provider_job_id == "job-123"
    assert str(request.input_video_url) == "https://example.com/input.mp4"


def test_runpod_handler_returns_normalized_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_job(request, provider_job_id=None) -> DubbingJobResult:  # noqa: ANN001
        return DubbingJobResult(
            output_video_url=request.output_video_url,
            seed=request.seed,
            input_frames=121,
            stage1_width=request.width,
            stage1_height=request.height,
            final_width=request.width * 2,
            final_height=request.height * 2,
            frame_rate=request.frame_rate,
            duration_seconds=12.5,
            provider_job_id=provider_job_id,
        )

    monkeypatch.setattr("ltx_service.providers.runpod_handler.run_job", fake_run_job)

    payload = handler(
        {
            "id": "job-123",
            "input": {
                "input_video_url": "https://example.com/input.mp4",
                "output_video_url": "https://example.com/output.mp4",
                "prompt": "The man is speaking English, saying: 'Hello!'",
            },
        }
    )

    assert payload["provider_job_id"] == "job-123"
    assert payload["final_width"] == 1536
