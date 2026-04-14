# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

from contextlib import suppress
from typing import Any, cast

import numpy as np
import polyscope.imgui as psim
from OpenGL import GL


class NumpyImageTexture:
    """Upload NumPy image data to an OpenGL texture for ImGui display."""

    def __init__(self) -> None:
        self._texture_id: int | None = None
        self._texture_ref: Any = None
        self._size: tuple[int, int] | None = None

    def update(self, image: np.ndarray) -> None:
        rgba = self._normalize_image(image)
        height, width = rgba.shape[:2]
        if self._texture_id is None or self._size != (width, height):
            self.release()
            self._texture_id = int(GL.glGenTextures(1))
            self._texture_ref = self._make_texture_ref(self._texture_id)
            self._size = (width, height)
            self._bind()
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                width,
                height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                rgba,
            )
        else:
            self._bind()
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                rgba,
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def texture_ref(self) -> Any:
        if self._texture_ref is None:
            raise RuntimeError("Texture has not been initialized")
        return self._texture_ref

    def size(self) -> tuple[int, int]:
        if self._size is None:
            raise RuntimeError("Texture has not been initialized")
        return self._size

    def release(self) -> None:
        if self._texture_id is None:
            return
        with suppress(Exception):
            GL.glDeleteTextures(int(self._texture_id))
        self._texture_id = None
        self._texture_ref = None
        self._size = None

    def _bind(self) -> None:
        if self._texture_id is None:
            raise RuntimeError("Texture has not been initialized")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

    @staticmethod
    def _make_texture_ref(texture_id: int) -> Any:
        psim_any = cast(Any, psim)
        return psim_any.ImTextureRef(texture_id)

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected image with shape (H, W, C), got {arr.shape}")

        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        channels = arr.shape[2]
        if channels == 1:
            rgb = np.repeat(arr, 3, axis=2)
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate((rgb, alpha), axis=2)
        elif channels == 3:
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate((arr, alpha), axis=2)
        elif channels == 4:
            rgba = arr
        else:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")

        return np.ascontiguousarray(rgba)
