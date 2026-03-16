import argparse
import json
import sys
from pathlib import Path

from ltx_service.bootstrap import bootstrap_models, ensure_model_paths
from ltx_service.config import RuntimeConfig, expected_model_paths
from ltx_service.runner import get_runtime, run_job
from ltx_service.schemas import DubbingJobRequest


def _bootstrap_command(args: argparse.Namespace) -> int:
    model_paths = bootstrap_models(
        Path(args.model_root),
        token=args.hf_token,
        force=args.force,
    )
    sys.stdout.write(json.dumps({key: str(value) for key, value in model_paths.__dict__.items()}, indent=2) + "\n")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    config = RuntimeConfig.from_env()
    if args.model_root:
        config = RuntimeConfig(
            model_root=Path(args.model_root).expanduser().resolve(),
            scratch_root=config.scratch_root,
            request_connect_timeout_seconds=config.request_connect_timeout_seconds,
            request_read_timeout_seconds=config.request_read_timeout_seconds,
            request_chunk_size=config.request_chunk_size,
            model_paths=expected_model_paths(Path(args.model_root).expanduser().resolve()),
        )
    ensure_model_paths(config.model_paths)
    get_runtime(config)
    sys.stdout.write(json.dumps({"status": "ready", "gpu": "cuda"}) + "\n")
    return 0


def _run_job_command(args: argparse.Namespace) -> int:
    with Path(args.request_json).open() as handle:
        payload = json.load(handle)
    request = DubbingJobRequest.model_validate(payload)
    result = run_job(request, provider_job_id=args.provider_job_id)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ltx-service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Download and verify required model artifacts.")
    bootstrap_parser.add_argument("--model-root", default="/models")
    bootstrap_parser.add_argument("--hf-token", default=None)
    bootstrap_parser.add_argument("--force", action="store_true")
    bootstrap_parser.set_defaults(func=_bootstrap_command)

    validate_parser = subparsers.add_parser("validate-runtime", help="Verify model files and CUDA availability.")
    validate_parser.add_argument("--model-root", default=None)
    validate_parser.set_defaults(func=_validate_command)

    run_job_parser = subparsers.add_parser("run-job", help="Run a local dubbing job from a JSON request payload.")
    run_job_parser.add_argument("--request-json", required=True)
    run_job_parser.add_argument("--provider-job-id", default=None)
    run_job_parser.set_defaults(func=_run_job_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
