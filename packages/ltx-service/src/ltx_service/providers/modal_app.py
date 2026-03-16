import inspect
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ltx_service.runner import run_job
from ltx_service.schemas import DubbingJobRequest

try:
    import modal
except ImportError:  # pragma: no cover - optional at import time in local tests
    modal = None

ModalSubmitter = Callable[[dict[str, Any]], str | Awaitable[str]]
ModalPoller = Callable[[str], tuple[str, dict[str, Any] | None] | Awaitable[tuple[str, dict[str, Any] | None]]]
ResolvedValue = TypeVar("ResolvedValue")


async def _resolve_maybe_awaitable(value: ResolvedValue | Awaitable[ResolvedValue]) -> ResolvedValue:
    if inspect.isawaitable(value):
        return await value
    return value


def create_modal_web_app(*, submitter: ModalSubmitter, poller: ModalPoller) -> FastAPI:
    app = FastAPI(title="JustDubit Modal API")

    @app.post("/submit")
    async def submit(request: DubbingJobRequest) -> JSONResponse:
        call_id = await _resolve_maybe_awaitable(submitter(request.model_dump(mode="json")))
        return JSONResponse(
            status_code=202,
            content={
                "status": "submitted",
                "call_id": call_id,
            },
        )

    @app.get("/result/{call_id}")
    async def result(call_id: str) -> JSONResponse:
        status, payload = await _resolve_maybe_awaitable(poller(call_id))
        if status == "pending":
            return JSONResponse(status_code=202, content={"status": "pending", "call_id": call_id})
        if status == "expired":
            return JSONResponse(status_code=404, content={"status": "expired", "call_id": call_id})
        if status == "error":
            return JSONResponse(status_code=500, content={"status": "error", "call_id": call_id, "result": payload})
        return JSONResponse(status_code=200, content={"status": "completed", "call_id": call_id, "result": payload})

    return app


def _dockerfile_path() -> str:
    return str(Path(__file__).resolve().parents[5] / "docker" / "serverless.Dockerfile")


if modal is not None:  # pragma: no branch
    app = modal.App(os.getenv("LTX_MODAL_APP_NAME", "justdubit-serverless"))
    volume = modal.Volume.from_name(os.getenv("LTX_MODAL_VOLUME_NAME", "justdubit-models"), create_if_missing=True)

    image_ref = os.getenv("LTX_SERVICE_IMAGE_URI")
    if image_ref:
        image = modal.Image.from_registry(image_ref, add_python="3.12")
    else:
        image = modal.Image.from_dockerfile(_dockerfile_path())

    @app.function(
        image=image,
        gpu=os.getenv("LTX_MODAL_GPU", "H100"),
        volumes={"/models": volume},
        timeout=int(os.getenv("LTX_MODAL_TIMEOUT_SECONDS", "7200")),
        startup_timeout=int(os.getenv("LTX_MODAL_STARTUP_TIMEOUT_SECONDS", "3600")),
        min_containers=0,
    )
    def run_modal_job(payload: dict[str, Any]) -> dict[str, Any]:
        request = DubbingJobRequest.model_validate(payload)
        result = run_job(request)
        return result.model_dump(mode="json")

    def _modal_call_id(function_call: object) -> str:
        for attribute in ("object_id", "function_call_id", "id"):
            value = getattr(function_call, attribute, None)
            if value:
                return str(value)
        raise RuntimeError("Modal spawn returned a call without an identifier")

    async def _spawn_modal_job(payload: dict[str, Any]) -> str:
        function_call = run_modal_job.spawn(payload)
        return _modal_call_id(function_call)

    def _poll_modal_call(call_id: str) -> tuple[str, dict[str, Any] | None]:
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            result = function_call.get(timeout=0)
            if not isinstance(result, dict):
                result = {"value": result}
            return "completed", result
        except TimeoutError:
            return "pending", None
        except Exception as exc:  # pragma: no cover - exercised only with real Modal runtime
            if type(exc).__name__ == "OutputExpiredError":
                return "expired", None
            return "error", {"error_code": type(exc).__name__, "error_message": str(exc)}

    @app.function(image=image)
    @modal.asgi_app()
    def modal_api() -> FastAPI:
        return create_modal_web_app(submitter=_spawn_modal_job, poller=_poll_modal_call)
