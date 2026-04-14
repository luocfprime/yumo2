# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

import numpy as np

from yumo2.texture import (
    _barycentric_coordinates,
    apply_denoise,
    bake_texture,
    build_face_map,
    pad_texture_edges,
    sample_texture_positions,
    trilinear_sample,
)


def test_trilinear_sample_is_exact_for_linear_field() -> None:
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([0.0, 1.0])

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    scalar_field = xx + yy + zz

    positions = np.array([
        [0.25, 0.5, 0.75],
        [1.0, 1.0, 1.0],
    ])

    samples = trilinear_sample(positions, scalar_field, xs, ys, zs)

    assert np.allclose(samples, [1.5, 3.0])


def test_build_face_map_marks_covered_texels() -> None:
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    face_map = build_face_map(uvs, faces, 4, 4)

    assert face_map.dtype == np.int32
    assert np.any(face_map == 0)
    assert np.any(face_map == -1)


def test_bake_texture_rasterizes_triangle_values() -> None:
    vertices_unwrapped = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces_unwrapped = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([0.0, 1.0])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    scalar_field = xx + yy + zz

    texture = bake_texture(vertices_unwrapped, faces_unwrapped, uvs, scalar_field, xs, ys, zs, 4, 4)

    assert texture.shape == (4, 4)
    assert np.count_nonzero(texture) > 0


def test_apply_denoise_respects_mask() -> None:
    texture = np.zeros((5, 5), dtype=float)
    texture[2, 2] = 1.0
    uv_mask = np.zeros((5, 5), dtype=float)
    uv_mask[1:4, 1:4] = 1.0

    denoised = apply_denoise(texture, uv_mask, sigma=1.0)

    assert denoised[2, 2] < 1.0
    assert denoised[0, 0] == 0.0


def test_pad_texture_edges_extends_nearest_valid_values_into_padding() -> None:
    texture = np.array([
        [10.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 20.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    uv_mask = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    padded = pad_texture_edges(texture, uv_mask, iterations=1)

    assert np.array_equal(padded[uv_mask > 0], texture[uv_mask > 0])
    assert padded[0, 1] in {10.0, 20.0}
    assert padded[1, 0] in {10.0, 20.0}
    assert padded[4, 4] == 0.0


def test_barycentric_coordinates_return_nan_for_degenerate_triangles() -> None:
    points = np.array([[0.25, 0.25]])
    triangles = np.array([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]])

    barycentric = _barycentric_coordinates(points, triangles)

    assert barycentric.shape == (1, 3)
    assert np.all(np.isnan(barycentric))


def test_sample_texture_positions_skips_degenerate_uv_triangles() -> None:
    vertices_unwrapped = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces_unwrapped = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    face_map, rows, cols, positions = sample_texture_positions(vertices_unwrapped, faces_unwrapped, uvs, 4, 4)

    assert np.any(face_map == 0)
    assert len(rows) == 0
    assert len(cols) == 0
    assert positions.shape == (0, 3)
