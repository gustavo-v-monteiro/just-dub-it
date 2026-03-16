import av
import cv2
import torch

import ltx_pipelines


def test_import_smoke() -> None:
    assert av is not None
    assert cv2 is not None
    assert ltx_pipelines is not None
    assert torch is not None
