# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import structlog
import trimesh

logger = structlog.get_logger(__name__)


def _numeric_lines(path: Path):
    for line in path.open():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            [float(part) for part in parts]
        except ValueError:
            continue
        yield line


def _validate_uniform_spacing(axis: np.ndarray, axis_name: str) -> None:
    if axis.size <= 2:
        return

    deltas = np.diff(axis)
    # Accept small decimal quantization noise from text exports such as
    # 1/30 being written as alternating 0.033333 and 0.033334.
    # TODO: This tolerance is hard-coded for now.
    reference_delta = float(np.median(deltas))
    tolerance = max(1e-8, abs(reference_delta) * 5e-5)
    if np.max(np.abs(deltas - reference_delta)) > tolerance:
        raise ValueError(f"Axis {axis_name} does not have uniform spacing")


def validate_meshgrid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate that *points* form a regular 3-D grid and build the scalar field array.

    Args:
        points: Shape ``(N, 4)`` array of ``[x, y, z, value]`` rows.

    Returns:
        ``(scalar_field, xs, ys, zs)`` where *scalar_field* has shape
        ``(len(xs), len(ys), len(zs))`` and NaN for any missing grid cell.

    Raises:
        ValueError: If the data is not shaped ``(N, 4)``, axes are non-uniform,
            the point count exceeds the grid, or duplicate grid cells are detected.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected points with shape (N, 4), got {points.shape}")

    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    zs = np.unique(points[:, 2])

    _validate_uniform_spacing(xs, "x")
    _validate_uniform_spacing(ys, "y")
    _validate_uniform_spacing(zs, "z")

    xi = np.searchsorted(xs, points[:, 0])
    yi = np.searchsorted(ys, points[:, 1])
    zi = np.searchsorted(zs, points[:, 2])

    scalar_field = np.full((len(xs), len(ys), len(zs)), np.nan, dtype=np.float64)

    if len(points) > scalar_field.size:
        raise ValueError("Point count exceeds meshgrid size")

    # Detect duplicate grid cells: same (xi, yi, zi) appearing more than once
    flat_indices = xi * (len(ys) * len(zs)) + yi * len(zs) + zi
    if len(flat_indices) != len(np.unique(flat_indices)):
        raise ValueError("Duplicate grid cells detected in input data")

    scalar_field[xi, yi, zi] = points[:, 3]

    missing_cells = int(np.isnan(scalar_field).sum())
    if missing_cells:
        logger.warning("input_scalar_field_has_missing_grid_cells", missing_cells=missing_cells)

    return scalar_field, xs, ys, zs


def load_scalar_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a scalar field from a plain-text or Tecplot ``.plt`` file.

    Each data line must contain exactly four whitespace-separated floats
    ``x y z value``.  Header/comment lines are skipped automatically.

    Args:
        path: Path to the data file.

    Returns:
        ``(scalar_field, xs, ys, zs)`` — see :func:`validate_meshgrid`.

    Raises:
        ValueError: If no valid numeric rows are found or the data is not a
            regular grid.
    """
    gen = _numeric_lines(path)
    first = next(gen, None)
    if first is None:
        raise ValueError(f"No numeric scalar field rows found in {path}")

    points = np.loadtxt(itertools.chain([first], gen), dtype=np.float64)
    points = np.atleast_2d(points)

    return validate_meshgrid(points)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangular mesh from *path* using trimesh.

    Args:
        path: Path to a mesh file (``.stl``, ``.obj``, ``.ply``, …).

    Returns:
        ``(vertices, faces)`` as float64 and int arrays respectively.

    Raises:
        TypeError: If the file does not yield a triangular mesh.
    """
    mesh = trimesh.load_mesh(path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Could not load a triangular mesh from {path}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)
