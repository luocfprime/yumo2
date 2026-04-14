# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import cv2
import numpy as np
import structlog
import xatlas

logger = structlog.get_logger(__name__)


def unwrap_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    padding: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """UV-unwrap a triangular mesh using xatlas.

    Args:
        vertices: ``(V, 3)`` vertex positions.
        faces: ``(F, 3)`` face indices.
        padding: Per-chart padding in texels (prevents UV bleeding between islands).

    Returns:
        ``(uvs, faces_unwrapped, vertices_unwrapped, param_corner, atlas_height, atlas_width)``

        - *uvs*: ``(V', 2)`` UV coordinates in ``[0, 1]²``.
        - *faces_unwrapped*: ``(F, 3)`` face indices into *uvs* / *vertices_unwrapped*.
        - *vertices_unwrapped*: ``(V', 3)`` vertices reindexed to match the UV atlas.
        - *param_corner*: ``(F*3, 2)`` per-corner UVs (convenience reshape).
        - *atlas_height*, *atlas_width*: atlas texture dimensions in texels.
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)

    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.bilinear = True
    pack_options.rotate_charts = True

    atlas.generate(pack_options=pack_options)

    vmapping, faces_unwrapped, uvs = atlas[0]
    vertices_unwrapped = vertices[vmapping]
    param_corner = uvs[faces_unwrapped].reshape(-1, 2)

    return (
        uvs,
        faces_unwrapped,
        vertices_unwrapped,
        param_corner,
        atlas.height,
        atlas.width,
    )


def build_face_map(uvs: np.ndarray, faces_unwrapped: np.ndarray, height: int, width: int) -> np.ndarray:
    uv_pixels = np.empty_like(uvs, dtype=np.float64)
    uv_pixels[:, 0] = uvs[:, 0] * (width - 1)
    uv_pixels[:, 1] = (1.0 - uvs[:, 1]) * (height - 1)

    face_map = np.zeros((height, width), dtype=np.int32)
    for face_index, face in enumerate(faces_unwrapped):
        polygon = np.round(uv_pixels[face]).astype(np.int32)
        cv2.fillConvexPoly(face_map, polygon.reshape((-1, 1, 2)), face_index + 1)

    return face_map - 1


def _barycentric_coordinates(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    denom = d00 * d11 - d01 * d01
    bary_b = np.full(len(points), np.nan, dtype=np.float64)
    bary_c = np.full(len(points), np.nan, dtype=np.float64)

    valid = np.abs(denom) > 1e-14
    bary_b[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    bary_c[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    bary_a = 1.0 - bary_b - bary_c

    return np.stack([bary_a, bary_b, bary_c], axis=1)


def _axis_lerp(axis: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(axis) == 1:
        zeros = np.zeros(len(coords), dtype=np.int64)
        return zeros, zeros, np.zeros(len(coords), dtype=np.float64)

    clipped = np.clip(coords, axis[0], axis[-1])
    upper = np.searchsorted(axis, clipped, side="right")
    upper = np.clip(upper, 1, len(axis) - 1)
    lower = upper - 1

    lower_values = axis[lower]
    upper_values = axis[upper]
    weights = (clipped - lower_values) / (upper_values - lower_values)

    return lower, upper, weights


def trilinear_sample(
    positions: np.ndarray,
    scalar_field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Trilinearly interpolate *scalar_field* at arbitrary 3-D *positions*.

    Positions outside the grid are clamped to the grid boundary.

    Args:
        positions: ``(N, 3)`` query coordinates.
        scalar_field: ``(Nx, Ny, Nz)`` volumetric data array.
        xs, ys, zs: 1-D axis coordinate arrays defining the grid.

    Returns:
        ``(N,)`` interpolated values.
    """
    x0, x1, tx = _axis_lerp(xs, positions[:, 0])
    y0, y1, ty = _axis_lerp(ys, positions[:, 1])
    z0, z1, tz = _axis_lerp(zs, positions[:, 2])

    c000 = scalar_field[x0, y0, z0]
    c100 = scalar_field[x1, y0, z0]
    c010 = scalar_field[x0, y1, z0]
    c110 = scalar_field[x1, y1, z0]
    c001 = scalar_field[x0, y0, z1]
    c101 = scalar_field[x1, y0, z1]
    c011 = scalar_field[x0, y1, z1]
    c111 = scalar_field[x1, y1, z1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


def bake_texture(
    vertices_unwrapped: np.ndarray,
    faces_unwrapped: np.ndarray,
    uvs: np.ndarray,
    scalar_field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Bake one scalar sample per covered texel.

    The texture resolution `(height, width)` defines the surface sampling grid in
    UV space: each covered texel center is mapped back to 3D and sampled once.
    """
    face_map = build_face_map(uvs, faces_unwrapped, height, width)
    rows, cols = np.where(face_map >= 0)
    texture = np.zeros((height, width), dtype=np.float64)

    if len(rows) == 0:
        return texture

    face_indices = face_map[rows, cols]
    uv_triangles = uvs[faces_unwrapped[face_indices]]
    vertex_triangles = vertices_unwrapped[faces_unwrapped[face_indices]]

    sample_points = np.column_stack(((cols + 0.5) / width, 1.0 - (rows + 0.5) / height))
    barycentric = _barycentric_coordinates(sample_points, uv_triangles)
    positions = np.einsum("ij,ijk->ik", barycentric, vertex_triangles)
    values = trilinear_sample(positions, scalar_field, xs, ys, zs)

    texture[rows, cols] = values
    return texture


def sample_texture_positions(
    vertices_unwrapped: np.ndarray,
    faces_unwrapped: np.ndarray,
    uvs: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 3-D world position for every covered texel center.

    Used by :class:`~yumo2.features.scope.Scope` for spatial radius queries —
    each texel maps to one point on the mesh surface.

    Args:
        vertices_unwrapped: ``(V', 3)`` vertex positions reindexed by xatlas.
        faces_unwrapped: ``(F, 3)`` face indices into *vertices_unwrapped*.
        uvs: ``(V', 2)`` UV coordinates.
        height, width: Atlas texture dimensions.

    Returns:
        ``(face_map, rows, cols, positions)``

        - *face_map*: ``(H, W)`` int array mapping each texel to its face index
          (``-1`` for uncovered texels).
        - *rows*, *cols*: 1-D index arrays of covered texels.
        - *positions*: ``(N, 3)`` world-space position of each covered texel.
    """
    face_map = build_face_map(uvs, faces_unwrapped, height, width)
    rows, cols = np.where(face_map >= 0)

    if len(rows) == 0:
        return face_map, rows, cols, np.zeros((0, 3), dtype=np.float64)

    face_indices = face_map[rows, cols]
    uv_triangles = uvs[faces_unwrapped[face_indices]]
    vertex_triangles = vertices_unwrapped[faces_unwrapped[face_indices]]
    sample_points = np.column_stack(((cols + 0.5) / width, 1.0 - (rows + 0.5) / height))
    barycentric = _barycentric_coordinates(sample_points, uv_triangles)
    valid = np.all(np.isfinite(barycentric), axis=1)
    if not np.all(valid):
        logger.warning(
            "degenerate_uv_triangles_skipped",
            skipped_texels=int((~valid).sum()),
            total_texels=len(valid),
        )
        rows = rows[valid]
        cols = cols[valid]
        vertex_triangles = vertex_triangles[valid]
        barycentric = barycentric[valid]
    positions = np.einsum("ij,ijk->ik", barycentric, vertex_triangles)
    return face_map, rows, cols, positions


def apply_denoise(texture: np.ndarray, uv_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply masked Gaussian blur to *texture*, preserving UV island boundaries.

    Blending is weighted by *uv_mask* so that smoothing never bleeds across
    UV seams into uncovered texels.

    Args:
        texture: ``(H, W)`` raw baked texture (float64).
        uv_mask: ``(H, W)`` float mask; 1.0 inside covered texels, 0.0 outside.
        sigma: Gaussian standard deviation in texels.  ``sigma <= 0`` returns
            an unmodified copy.

    Returns:
        Smoothed ``(H, W)`` texture, zeroed outside the UV mask.
    """
    if sigma <= 0:
        return texture.copy()

    weighted_texture = texture * uv_mask
    blurred_values = cv2.GaussianBlur(weighted_texture, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred_weights = cv2.GaussianBlur(uv_mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    result = np.zeros_like(texture, dtype=np.float64)
    valid = blurred_weights > 1e-12
    result[valid] = blurred_values[valid] / blurred_weights[valid]
    result[uv_mask <= 0] = 0.0
    return result
