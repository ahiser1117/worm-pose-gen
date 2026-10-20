"""Images exchanged with the browser: frames, masks and probability maps as data URLs.

Frames go out as JPEG (``jpeg_data_url``: a full-resolution gray frame is
several times smaller than as PNG), everything the browser reads pixel by
pixel as PNG.  Mask PNGs use 0 = background, 255 = worm and 128 = ignore
(the label value 255, ``segmenter.IGNORE_LABEL``, which only migrated labels
still carry); a mask coming back from the browser is read with any mid-gray
as ignore, so a resampled edge never turns into worm.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..segmenter import IGNORE_LABEL

PNG_WORM = 255
PNG_IGNORE = 128


def encode_png(values: NDArray[np.generic]) -> bytes:
    """A uint8 gray, RGB or RGBA image as PNG bytes."""

    image = np.ascontiguousarray(values, dtype=np.uint8)
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4))):
        raise ValueError("PNG input must be a uint8 gray, RGB, or RGBA image")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def data_url(values: NDArray[np.generic]) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(values)).decode("ascii")


def jpeg_data_url(gray: NDArray[np.uint8], quality: int = 88) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(gray, dtype=np.uint8)).save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_data_url(mask: NDArray[np.generic]) -> str:
    """A boolean mask as a PNG data URL (0 / 255)."""

    return data_url(np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8))


def probability_to_png(probability: NDArray[np.floating]) -> NDArray[np.uint8]:
    return np.clip(np.rint(np.asarray(probability, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def probability_data_url(probability: NDArray[np.floating]) -> str:
    return data_url(probability_to_png(probability))


def mask_to_png_values(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Labels (0/1/255) to browser PNG values (0/255/128)."""

    values = np.asarray(mask)
    out = np.zeros(values.shape, dtype=np.uint8)
    out[values == 1] = PNG_WORM
    out[values == IGNORE_LABEL] = PNG_IGNORE
    return out


def png_values_to_mask(values: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Browser PNG values back to labels; anything mid-gray is ignore."""

    gray = np.asarray(values)
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[gray >= 192] = 1
    out[(gray > 64) & (gray < 192)] = IGNORE_LABEL
    return out


def decode_mask_data_url(url: str, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """A mask PNG data URL from the browser as labels; it must have the frame's ``shape``."""

    if not url.startswith("data:image/png;base64,"):
        raise ValueError("mask must be a base64 PNG data URL")
    raw = base64.b64decode(url.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("L")
    values = np.asarray(image, dtype=np.uint8)
    if values.shape != tuple(shape):
        raise ValueError(f"mask shape {values.shape} does not match frame {tuple(shape)}")
    return png_values_to_mask(values)
