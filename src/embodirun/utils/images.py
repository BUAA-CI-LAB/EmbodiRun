"""Strict bounded data-URL encoding used by camera and remote-model clients."""

from __future__ import annotations

import base64
import binascii

DEFAULT_MAX_IMAGE_BYTES = 16 * 1024 * 1024


class DataUrlError(ValueError):
    """An encoded image does not satisfy the declared media contract."""


def encode_data_url(data: bytes, media_type: str, *, maximum_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> str:
    if not isinstance(data, bytes) or not data:
        raise DataUrlError("data must be non-empty bytes")
    if len(data) > maximum_bytes:
        raise DataUrlError(f"data exceeds {maximum_bytes} bytes")
    if not isinstance(media_type, str) or not media_type.startswith("image/"):
        raise DataUrlError("media_type must be an image media type")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def decode_data_url(
    value: object,
    media_type: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> bytes:
    if not isinstance(value, str):
        raise DataUrlError("value must be a data URL")
    prefix = f"data:{media_type};base64,"
    if not value.startswith(prefix):
        raise DataUrlError(f"value must use {media_type}")
    encoded = value[len(prefix) :]
    maximum_encoded = ((maximum_bytes + 2) // 3) * 4
    if not encoded or len(encoded) > maximum_encoded:
        raise DataUrlError(f"decoded data exceeds {maximum_bytes} bytes")
    try:
        result = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DataUrlError("value contains invalid base64") from error
    if not result or len(result) > maximum_bytes:
        raise DataUrlError("value has an invalid decoded size")
    return result


__all__ = ["DEFAULT_MAX_IMAGE_BYTES", "DataUrlError", "decode_data_url", "encode_data_url"]
