from fastapi.testclient import TestClient

from ltx_service.providers.modal_app import create_modal_web_app


def test_modal_submit_and_result_endpoints() -> None:
    def submitter(_: dict[str, object]) -> str:
        return "call-123"

    def poller(call_id: str) -> tuple[str, dict[str, object]]:
        return (
            "completed",
            {
                "output_video_url": "https://example.com/output.mp4",
                "seed": 42,
                "input_frames": 121,
                "stage1_width": 768,
                "stage1_height": 512,
                "final_width": 1536,
                "final_height": 1024,
                "frame_rate": 24.0,
                "duration_seconds": 12.0,
                "provider_job_id": call_id,
                "error_code": None,
                "error_message": None,
                "error_details": None,
            },
        )

    app = create_modal_web_app(
        submitter=submitter,
        poller=poller,
    )
    client = TestClient(app)

    submit_response = client.post(
        "/submit",
        json={
            "input_video_url": "https://example.com/input.mp4",
            "output_video_url": "https://example.com/output.mp4",
            "prompt": "The man is speaking English, saying: 'Hello!'",
        },
    )
    assert submit_response.status_code == 202
    assert submit_response.json()["call_id"] == "call-123"

    result_response = client.get("/result/call-123")
    assert result_response.status_code == 200
    assert result_response.json()["result"]["provider_job_id"] == "call-123"


def test_modal_result_pending() -> None:
    def submitter(_: dict[str, object]) -> str:
        return "call-123"

    def poller(_: str) -> tuple[str, None]:
        return ("pending", None)

    app = create_modal_web_app(
        submitter=submitter,
        poller=poller,
    )
    client = TestClient(app)

    response = client.get("/result/call-123")

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
