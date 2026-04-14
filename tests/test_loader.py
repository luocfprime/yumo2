# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

import numpy as np
import pytest

from yumo2.loader import load_scalar_field, validate_meshgrid


def test_validate_meshgrid_returns_axes_and_scalar_field() -> None:
    points = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, 0.0, 3.0],
        [1.0, 2.0, 0.0, 4.0],
    ])

    scalar_field, xs, ys, zs = validate_meshgrid(points)

    assert scalar_field.shape == (2, 2, 1)
    assert xs.tolist() == [0.0, 1.0]
    assert ys.tolist() == [0.0, 2.0]
    assert zs.tolist() == [0.0]
    assert scalar_field[:, :, 0].tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_load_scalar_field_warns_and_fills_missing_cells(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "missing.plt"
    path.write_text(
        "\n".join([
            "variables = x, y, z, value",
            "0 0 0 1",
            "1 0 0 2",
            "0 1 0 3",
        ])
    )

    scalar_field, xs, ys, zs = load_scalar_field(path)

    assert scalar_field.shape == (2, 2, 1)
    assert xs.tolist() == [0.0, 1.0]
    assert ys.tolist() == [0.0, 1.0]
    assert zs.tolist() == [0.0]
    assert np.isnan(scalar_field[1, 1, 0])
    assert "input_scalar_field_has_missing_grid_cells" in caplog.text


def test_validate_meshgrid_rejects_irregular_spacing() -> None:
    points = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [3.0, 0.0, 0.0, 3.0],
    ])

    with pytest.raises(ValueError, match="uniform spacing"):
        validate_meshgrid(points)


def test_validate_meshgrid_allows_small_decimal_rounding_noise() -> None:
    points = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.033333, 2.0],
        [0.0, 0.0, 0.066667, 3.0],
        [0.0, 0.0, 0.1, 4.0],
    ])

    scalar_field, xs, ys, zs = validate_meshgrid(points)

    assert scalar_field.shape == (1, 1, 4)
    assert xs.tolist() == [0.0]
    assert ys.tolist() == [0.0]
    assert zs.tolist() == [0.0, 0.033333, 0.066667, 0.1]
