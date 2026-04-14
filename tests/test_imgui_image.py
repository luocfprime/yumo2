# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

import numpy as np

from yumo2.imgui_image import NumpyImageTexture


def test_normalize_image_adds_opaque_alpha_for_rgb_float() -> None:
    image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    rgba = NumpyImageTexture._normalize_image(image)

    assert rgba.dtype == np.uint8
    assert rgba.shape == (1, 1, 4)
    assert rgba[0, 0].tolist() == [0, 127, 255, 255]


def test_normalize_image_expands_single_channel_to_rgba() -> None:
    image = np.array([[64, 128]], dtype=np.uint8)

    rgba = NumpyImageTexture._normalize_image(image)

    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0].tolist() == [64, 64, 64, 255]
    assert rgba[0, 1].tolist() == [128, 128, 128, 255]
