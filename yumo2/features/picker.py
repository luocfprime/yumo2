# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from yumo2.profiling import freq_time_profiler
from yumo2.ui import ui_tree_node

if TYPE_CHECKING:
    from yumo2.app import PolyscopeApp


_PADDING = 4
logger = structlog.get_logger(__name__)


def _format_position(values: np.ndarray) -> str:
    coords = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{value:.3f}" for value in coords) + "]"


def _format_scientific(value: float) -> str:
    return f"{value:.6e}"


class Picker:
    """Point-and-click surface value query.

    On each left-click inside the "Coord Picker" panel, reports the 3-D world
    coordinate of the click and, when the click lands on the mesh, uses
    barycentric interpolation in UV space to look up both the raw and
    denoised scalar values at that surface point.
    """

    def __init__(self, app: PolyscopeApp) -> None:
        self.app = app

    def mesh_pick_values(self, face_index: int, barycentric_coords: np.ndarray) -> tuple[float, float | None, float]:
        uvs = self.app._session.uvs
        faces_unwrapped = self.app._session.faces_unwrapped
        original_texture = self.app._session.original_texture
        raw_texture = self.app._session.raw_texture
        texture = self.app._session.texture
        if uvs is None or faces_unwrapped is None:
            raise RuntimeError("Mesh pick requires UV state")
        if original_texture is None or raw_texture is None or texture is None:
            raise RuntimeError("Mesh pick requires baked textures")

        uv_triangle = uvs[faces_unwrapped[face_index]]
        uv = barycentric_coords @ uv_triangle

        texture_height, texture_width = texture.shape[:2]
        u, v = uv
        col = int(np.clip(u * (texture_width - 1), 0, texture_width - 1))
        row = int(np.clip((1.0 - v) * (texture_height - 1), 0, texture_height - 1))

        original_value = float(original_texture[row, col])
        surface_value = float(raw_texture[row, col])
        smoothed_value = None
        if self.app.settings.denoise_enabled and self.app.settings.denoise_sigma > 0:
            smoothed_value = float(texture[row, col])

        return surface_value, smoothed_value, original_value

    @freq_time_profiler("ui_picker")
    def ui(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self.app._psim, "Coord Picker", open_first_time=False) as expanded:
            if not expanded:
                return

            io = self.app._psim.GetIO()
            if io.MouseClicked[0]:
                self.app._session.picker_msgs = []

                screen_coords = io.MousePos
                world_coords = self.app._ps.screen_coords_to_world_position(screen_coords)
                self.app._session.picker_msgs.append(f"World coord: {_format_position(world_coords)}")
                logger.debug("picker_clicked", screen_coords=tuple(screen_coords), world_coords=tuple(world_coords))

                pick_result = self.app._ps.pick(screen_coords=screen_coords)
                if pick_result.is_hit and pick_result.structure_name == "mesh":
                    data = pick_result.structure_data
                    if "bary_coords" in data:
                        surface_value, smoothed_value, original_value = self.mesh_pick_values(
                            int(data["index"]),
                            np.asarray(data["bary_coords"], dtype=float),
                        )
                        self.app._session.picker_msgs.append(f"Surface value: {surface_value:.6g}")
                        if self.app.settings.scalar_transform in ("log_e", "log_10"):
                            self.app._session.picker_msgs.append(
                                f"Original surface value: {_format_scientific(original_value)}"
                            )
                        logger.debug(
                            "picker_sampled_values",
                            face_index=int(data["index"]),
                            surface_value=surface_value,
                            smoothed_value=smoothed_value,
                            original_value=original_value,
                        )

                        if smoothed_value is not None:
                            self.app._session.picker_msgs.append(f"Smoothed surface value: {smoothed_value:.6g}")
                            if surface_value != 0:
                                rel_error = abs(smoothed_value - surface_value) / abs(surface_value)
                                self.app._session.picker_msgs.append(f"Rel error: {rel_error:.6g}")
                                logger.debug("picker_relative_error", relative_error=rel_error)
                            else:
                                self.app._session.picker_msgs.append("Rel error: N/A (surface value is 0)")
                                logger.debug("picker_relative_error_unavailable", reason="surface_value_zero")

            for message in self.app._session.picker_msgs:
                self.app._psim.Text(message)

            for _ in range(max(0, _PADDING - len(self.app._session.picker_msgs))):
                self.app._psim.Text("")
