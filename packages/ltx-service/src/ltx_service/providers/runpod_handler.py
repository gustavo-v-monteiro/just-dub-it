from typing import Any

import runpod

from ltx_service.runner import run_job
from ltx_service.schemas import DubbingJobRequest


def parse_runpod_event(event: dict[str, Any]) -> tuple[DubbingJobRequest, str | None]:
    payload = event.get("input", event)
    provider_job_id = event.get("id")
    request = DubbingJobRequest.model_validate(payload)
    return request, str(provider_job_id) if provider_job_id is not None else None


def handler(event: dict[str, Any]) -> dict[str, Any]:
    request, provider_job_id = parse_runpod_event(event)
    result = run_job(request, provider_job_id=provider_job_id)
    return result.model_dump(mode="json")


def main() -> None:
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
