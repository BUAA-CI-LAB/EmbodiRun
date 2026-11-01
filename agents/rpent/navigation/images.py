"""Compress only planner feedback; policy observations retain their original RGB."""

from __future__ import annotations

import io
from dataclasses import replace

JPEG_QUALITY = 80


def planner_image(feedback):
    if feedback.mime_type == "image/jpeg":
        return feedback
    from PIL import Image

    with Image.open(io.BytesIO(feedback.image)) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return replace(feedback, image=buffer.getvalue(), mime_type="image/jpeg")
