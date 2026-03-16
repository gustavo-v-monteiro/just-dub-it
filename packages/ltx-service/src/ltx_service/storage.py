from pathlib import Path

import requests


def download_to_path(
    url: str,
    destination: Path,
    *,
    timeout: tuple[int, int],
    chunk_size: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def upload_file(
    url: str,
    source: Path,
    *,
    timeout: tuple[int, int],
) -> None:
    with source.open("rb") as handle:
        response = requests.put(url, data=handle, timeout=timeout)
    response.raise_for_status()
