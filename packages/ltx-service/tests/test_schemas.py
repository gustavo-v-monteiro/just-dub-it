import pytest

from ltx_service.schemas import DubbingJobRequest


def test_request_defaults_and_validation() -> None:
    request = DubbingJobRequest.model_validate(
        {
            "input_video_url": "https://example.com/input.mp4",
            "output_video_url": "https://example.com/output.mp4",
            "prompt": "The man is speaking English, saying: 'Hello!'",
        }
    )

    assert request.height == 512
    assert request.width == 768
    assert request.conditioning_strength == 1.0


def test_request_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="multiples of 32"):
        DubbingJobRequest.model_validate(
            {
                "input_video_url": "https://example.com/input.mp4",
                "output_video_url": "https://example.com/output.mp4",
                "prompt": "Hello",
                "height": 510,
            }
        )
