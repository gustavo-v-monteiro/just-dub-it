from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import pytest
import requests

from ltx_service.storage import download_to_path, upload_file


class _FakeResponse:
    def __init__(self, *, chunks: list[bytes] | None = None):
        self._chunks = chunks or []

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield from self._chunks


def test_download_to_path_writes_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destination = tmp_path / "input.mp4"

    def fake_get(url: str, stream: bool, timeout: tuple[int, int]) -> _FakeResponse:
        assert url == "https://example.com/input.mp4"
        assert stream is True
        assert timeout == (30, 60)
        return _FakeResponse(chunks=[b"abc", b"123"])

    monkeypatch.setattr(requests, "get", fake_get)

    download_to_path("https://example.com/input.mp4", destination, timeout=(30, 60), chunk_size=1024)

    assert destination.read_bytes() == b"abc123"


def test_upload_file_puts_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "output.mp4"
    source.write_bytes(b"video")
    captured: dict[str, bytes] = {}

    class _PutResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_put(url: str, data: BinaryIO, timeout: tuple[int, int]) -> _PutResponse:
        del timeout
        captured["url"] = url.encode()
        captured["body"] = data.read()
        return _PutResponse()

    monkeypatch.setattr(requests, "put", fake_put)

    upload_file("https://example.com/output.mp4", source, timeout=(30, 60))

    assert captured["url"].decode() == "https://example.com/output.mp4"
    assert captured["body"] == b"video"
