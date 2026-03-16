#!/usr/bin/env python3
"""Create or update the Runpod resources needed for the JustDubit serverless worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests
import runpod

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_WORKERS_MAX = 1
DEFAULT_GPU_COUNT = 1


@dataclass(frozen=True)
class DeployConfig:
    api_key: str
    image_name: str
    endpoint_name: str
    template_name: str
    network_volume_name: str
    network_volume_size_gb: int
    data_center_id: str
    allowed_cuda_versions: str
    gpu_priority: tuple[str, ...]
    workers_min: int
    workers_max: int
    idle_timeout: int
    scaler_value: int


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _rest_request(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=f"{RUNPOD_REST_BASE}{path}",
        headers=_auth_headers(api_key),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def _get_or_create_network_volume(config: DeployConfig) -> dict[str, Any]:
    user = runpod.get_user(api_key=config.api_key)
    for volume in user.get("networkVolumes", []):
        if volume["name"] == config.network_volume_name:
            return volume

    created = _rest_request(
        "POST",
        "/networkvolumes",
        api_key=config.api_key,
        payload={
            "name": config.network_volume_name,
            "size": config.network_volume_size_gb,
            "dataCenterId": config.data_center_id,
        },
    )
    return created


def _get_gpu_id_by_keyword(api_key: str, keyword: str) -> str:
    normalized_keyword = keyword.lower()
    matches = []
    for gpu in runpod.get_gpus(api_key=api_key):
        display_name = gpu.get("displayName") or gpu.get("id") or ""
        if normalized_keyword in display_name.lower() or normalized_keyword in str(gpu.get("id", "")).lower():
            matches.append(gpu)
    if not matches:
        raise ValueError(f"Could not find a GPU matching {keyword!r}")
    secure_first = sorted(matches, key=lambda item: (item.get("secureCloud", False) is False, item.get("id", "")))
    return secure_first[0]["id"]


def _select_gpu_ids(config: DeployConfig) -> str:
    gpu_ids = [_get_gpu_id_by_keyword(config.api_key, keyword) for keyword in config.gpu_priority]
    return ",".join(gpu_ids)


def _upsert_template(config: DeployConfig) -> dict[str, Any]:
    existing_templates = _rest_request("GET", "/templates", api_key=config.api_key).get("data", [])
    for template in existing_templates:
        if template.get("name") == config.template_name:
            return template

    return runpod.create_template(
        name=config.template_name,
        image_name=config.image_name,
        docker_start_cmd="python -m ltx_service.providers.runpod_handler",
        container_disk_in_gb=50,
        volume_mount_path="/runpod-volume",
        env={
            "LTX_MODEL_ROOT": "/runpod-volume/models",
            "LTX_SCRATCH_ROOT": "/tmp/jobs",
            "LTX_PROVIDER": "runpod",
        },
        is_serverless=True,
    )


def _find_endpoint_by_name(api_key: str, name: str) -> dict[str, Any] | None:
    endpoints = _rest_request("GET", "/serverless/endpoints", api_key=api_key).get("data", [])
    for endpoint in endpoints:
        if endpoint.get("name") == name:
            return endpoint
    return None


def _create_endpoint(config: DeployConfig, template_id: str, network_volume_id: str) -> dict[str, Any]:
    existing = _find_endpoint_by_name(config.api_key, config.endpoint_name)
    if existing is not None:
        return existing

    gpu_ids = _select_gpu_ids(config)
    return runpod.create_endpoint(
        name=config.endpoint_name,
        template_id=template_id,
        gpu_ids=gpu_ids,
        network_volume_id=network_volume_id,
        idle_timeout=config.idle_timeout,
        scaler_type="QUEUE_DELAY",
        scaler_value=config.scaler_value,
        workers_min=config.workers_min,
        workers_max=config.workers_max,
        allowed_cuda_versions=config.allowed_cuda_versions,
        gpu_count=DEFAULT_GPU_COUNT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--endpoint-name", default="justdubit-serverless")
    parser.add_argument("--template-name", default="justdubit-serverless-template")
    parser.add_argument("--network-volume-name", default="justdubit-models")
    parser.add_argument("--network-volume-size-gb", type=int, default=400)
    parser.add_argument("--data-center-id", default="EUR-IS-1")
    parser.add_argument("--allowed-cuda-versions", default="12.8")
    parser.add_argument("--gpu-priority", nargs="+", default=("H100", "A100 80GB"))
    parser.add_argument("--workers-min", type=int, default=0)
    parser.add_argument("--workers-max", type=int, default=DEFAULT_WORKERS_MAX)
    parser.add_argument("--idle-timeout", type=int, default=5)
    parser.add_argument("--scaler-value", type=int, default=4)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.api_key:
        parser.error("RUNPOD_API_KEY or --api-key is required")

    config = DeployConfig(
        api_key=args.api_key,
        image_name=args.image_name,
        endpoint_name=args.endpoint_name,
        template_name=args.template_name,
        network_volume_name=args.network_volume_name,
        network_volume_size_gb=args.network_volume_size_gb,
        data_center_id=args.data_center_id,
        allowed_cuda_versions=args.allowed_cuda_versions,
        gpu_priority=tuple(args.gpu_priority),
        workers_min=args.workers_min,
        workers_max=args.workers_max,
        idle_timeout=args.idle_timeout,
        scaler_value=args.scaler_value,
    )

    volume = _get_or_create_network_volume(config)
    template = _upsert_template(config)
    endpoint = _create_endpoint(config, template_id=template["id"], network_volume_id=volume["id"])
    sys.stdout.write(
        json.dumps(
            {
                "network_volume": volume,
                "template": template,
                "endpoint": endpoint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
