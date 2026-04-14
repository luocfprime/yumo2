# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

import numpy as np
import pytest

from yumo2.colormap import generate_colorbar_image, generate_colorbar_rgba, load_colormaps


@pytest.mark.parametrize(
    ("orientation", "size", "lhs", "rhs"),
    [
        ("vertical", (120, 80), (20, 20), (100, 20)),
        ("horizontal", (80, 120), (20, 20), (20, 100)),
    ],
)
def test_generate_colorbar_image_varies_along_orientation_axis(
    orientation: str,
    size: tuple[int, int],
    lhs: tuple[int, int],
    rhs: tuple[int, int],
) -> None:
    image = generate_colorbar_image(
        size[0],
        size[1],
        "viridis",
        0.0,
        1.0,
        orientation=orientation,
    )

    if orientation == "vertical":
        assert image.shape[0] == size[0]
    else:
        assert image.shape[1] == size[1]
    assert not np.allclose(image[lhs], image[rhs])


def test_generate_colorbar_rgba_has_transparent_background() -> None:
    image = generate_colorbar_rgba(
        120,
        80,
        "viridis",
        0.0,
        1.0,
        orientation="vertical",
    )

    assert image.shape[2] == 4
    assert image[0, 0, 3] == 0.0
    assert np.max(image[:, :, 3]) == 1.0


def test_load_colormaps_skips_already_loaded_names(tmp_path, monkeypatch) -> None:
    import yumo2.colormap as colormap_module

    monkeypatch.setattr(colormap_module, "_LOADED_PS_COLORMAPS", set())
    colormap_path = tmp_path / "magma_colormap.png"
    colormap_path.write_bytes(b"png")

    class FakePs:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def load_color_map(self, name: str, path: str) -> None:
            self.calls.append((name, path))

    ps = FakePs()

    first = load_colormaps(ps, tmp_path)
    second = load_colormaps(ps, tmp_path)

    assert first == {"magma": str(colormap_path)}
    assert second == {"magma": str(colormap_path)}
    assert ps.calls == [("magma", str(colormap_path))]
