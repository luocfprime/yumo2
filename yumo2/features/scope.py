# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog

from yumo2.markers import cross_nodes_edges
from yumo2.profiling import freq_time_profiler
from yumo2.ui import ui_equal_widths, ui_labeled_checkbox, ui_labeled_item_width, ui_select, ui_tree_node

if TYPE_CHECKING:
    from yumo2.app import PolyscopeApp


_PADDING = 4
_LINE_MAX_CROSS_FRACTION = 0.08

logger = structlog.get_logger(__name__)


def _format_position(values: np.ndarray) -> str:
    coords = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{value:.3f}" for value in coords) + "]"


def _format_scientific(value: float) -> str:
    return f"{value:.6e}"


@dataclass(frozen=True)
class ScopeQueryResult:
    position: np.ndarray
    raw_value: float
    smoothed_value: float | None
    original_value: float


class Scope:
    """Interactive spatial extremum finder.

    When enabled, a mouse click registers a world-space center point.  The
    sphere marker shows the query radius; on release, :meth:`query_extremum`
    searches all texel positions within the radius and returns the texel with
    the minimum or maximum raw scalar value.  A screen-space-proportional cross
    marker is placed at the result and updated every frame as the camera moves.
    """

    def __init__(self, app: PolyscopeApp) -> None:
        self.app = app
        self._sphere: Any = None
        self._cross: Any = None

    def cleanup(self) -> None:  # pragma: no cover - Polyscope callback
        self._hide_sphere()
        self._hide_cross()

    def query_extremum(
        self,
        center: np.ndarray,
        radius: float,
        mode: Literal["Min", "Max"],
    ) -> ScopeQueryResult | None:
        raw_texture = self.app._session.raw_texture
        original_texture = self.app._session.original_texture
        texture = self.app._session.texture
        rows = self.app._session.texel_rows
        cols = self.app._session.texel_cols
        positions = self.app._session.texel_positions
        if original_texture is None or raw_texture is None or texture is None:
            raise RuntimeError("Scope query requires baked textures")
        if rows is None or cols is None or positions is None:
            raise RuntimeError("Scope query requires sampled scope positions")

        distances = np.linalg.norm(positions - np.asarray(center, dtype=float), axis=1)
        inside = distances <= float(radius)
        if not np.any(inside):
            return None

        rows_in = rows[inside]
        cols_in = cols[inside]
        positions_in = positions[inside]
        raw_values = raw_texture[rows_in, cols_in]
        choice = int(np.argmax(raw_values) if mode == "Max" else np.argmin(raw_values))
        smoothed_value = None
        if self.app.settings.denoise_enabled and self.app.settings.denoise_sigma > 0:
            smoothed_value = float(texture[rows_in[choice], cols_in[choice]])

        return ScopeQueryResult(
            position=np.asarray(positions_in[choice], dtype=float),
            raw_value=float(raw_values[choice]),
            smoothed_value=smoothed_value,
            original_value=float(original_texture[rows_in[choice], cols_in[choice]]),
        )

    def _scene_scale(self) -> float:
        vertices = self.app._session.vertices
        if vertices is None or len(vertices) == 0:
            return 1.0
        return float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def _show_sphere(self, center: np.ndarray) -> None:  # pragma: no cover - Polyscope callback
        point = np.asarray(center, dtype=float).reshape(1, 3)
        if self._sphere is None:
            self._sphere = self.app._ps.register_point_cloud(
                "scope_sphere",
                point,
                enabled=True,
                point_render_mode="sphere",
                color=(1.0, 1.0, 1.0),
                transparency=0.35,
            )
            self._sphere.set_radius(self.app.settings.scope_radius, relative=False)
        else:
            self._sphere.update_point_positions(point)
            self._sphere.set_radius(self.app.settings.scope_radius, relative=False)
            self._sphere.set_enabled(True)

    def _hide_sphere(self) -> None:  # pragma: no cover - Polyscope callback
        if self._sphere is not None:
            self._sphere.set_enabled(False)

    def _show_cross(self, center: np.ndarray) -> None:  # pragma: no cover - Polyscope callback
        self.app._session.scope_cross_center = np.asarray(center, dtype=float)
        self._update_cross_geometry()

    def _update_cross_geometry(self) -> None:  # pragma: no cover - Polyscope callback
        if self.app._session.scope_cross_center is None:
            return
        import math

        cam = self.app._ps.get_view_camera_parameters()
        dist = float(np.linalg.norm(self.app._session.scope_cross_center - np.asarray(cam.get_position())))
        if dist == 0:
            return
        world_per_pixel = (2.0 * dist * math.tan(math.radians(cam.get_fov_vertical_deg()) / 2.0)) / float(
            self.app._psim.GetIO().DisplaySize[1]
        )
        max_cross_length = self._scene_scale() * self.app.settings.scope_marker_max_fraction
        cross_length = min(self.app.settings.scope_marker_length_px * world_per_pixel, max_cross_length)
        line_radius = min(
            0.5 * self.app.settings.scope_marker_thickness_px * world_per_pixel,
            max_cross_length * _LINE_MAX_CROSS_FRACTION,
        )
        nodes, edges = cross_nodes_edges(self.app._session.scope_cross_center, cross_length)
        if self._cross is None:
            self._cross = self.app._ps.register_curve_network("scope_cross", nodes, edges, enabled=True)
            self._cross.set_color((0.0, 0.0, 0.0))
            self._cross.set_material("flat")
        else:
            self._cross.update_node_positions(nodes)
            self._cross.set_enabled(True)
        self._cross.set_radius(line_radius, relative=False)

    def _hide_cross(self) -> None:  # pragma: no cover - Polyscope callback
        if hasattr(self.app, "_session"):
            self.app._session.scope_cross_center = None
        if self._cross is not None:
            self._cross.set_enabled(False)

    def _handle_interaction(self) -> None:  # pragma: no cover - Polyscope callback
        io = self.app._psim.GetIO()
        if io.MouseClicked[0]:
            self.app._session.scope_press_world = np.asarray(
                self.app._ps.screen_coords_to_world_position(io.MousePos),
                dtype=float,
            )
            logger.debug(
                "scope_press_started",
                screen_coords=tuple(io.MousePos),
                center=tuple(float(value) for value in self.app._session.scope_press_world),
            )
            self._show_sphere(self.app._session.scope_press_world)
            self._hide_cross()

        if io.MouseDown[0] and self.app._session.scope_press_world is not None:
            self._show_sphere(self.app._session.scope_press_world)

        if io.MouseReleased[0] and self.app._session.scope_press_world is not None:
            center = self.app._session.scope_press_world.copy()
            self.app._session.scope_press_world = None
            self._hide_sphere()
            logger.debug(
                "scope_started",
                center=tuple(float(value) for value in center),
                radius=self.app.settings.scope_radius,
                mode=self.app.settings.scope_mode,
            )

            self.app._session.scope_msgs = [f"Scope center: {_format_position(center)}"]
            result = self.query_extremum(center, self.app.settings.scope_radius, self.app.settings.scope_mode)
            if result is None:
                self._hide_cross()
                self.app._session.scope_msgs.append("No sampled mesh points in scope")
                logger.debug("scope_no_samples", center=tuple(float(value) for value in center))
                return

            self._show_cross(result.position)
            self.app._session.scope_msgs.append(
                f"Scope {self.app.settings.scope_mode.lower()} position: {_format_position(result.position)}"
            )
            self.app._session.scope_msgs.append(f"Raw surface value: {result.raw_value:.6g}")
            if self.app.settings.scalar_transform in ("log_e", "log_10"):
                self.app._session.scope_msgs.append(
                    f"Original surface value: {_format_scientific(result.original_value)}"
                )
            if result.smoothed_value is not None:
                self.app._session.scope_msgs.append(f"Smoothed surface value: {result.smoothed_value:.6g}")
            logger.debug(
                "scope_completed",
                center=tuple(float(value) for value in center),
                position=tuple(float(value) for value in result.position),
                raw_value=result.raw_value,
                smoothed_value=result.smoothed_value,
                original_value=result.original_value,
            )

    def _ui_primary_controls(self) -> None:  # pragma: no cover - Polyscope callback
        changed, scope_enabled = ui_labeled_checkbox(
            self.app._psim,
            "Enabled",
            "##scope_enabled",
            self.app.settings.scope_enabled,
        )
        if changed:
            self.app._set_settings(scope_enabled=scope_enabled)
            if not scope_enabled:
                self.cleanup()

        self.app._psim.SameLine()
        mode_width, radius_width = ui_equal_widths(self.app._psim, 2, min_width=50.0, max_width=100.0)
        with ui_labeled_item_width(self.app._psim, "Mode", mode_width):
            mode_changed, scope_mode = ui_select(
                self.app._psim,
                "##scope_mode",
                self.app.settings.scope_mode,
                ["Min", "Max"],
            )
        if mode_changed:
            self.app._set_settings(scope_mode=scope_mode)

        self.app._psim.SameLine()
        with ui_labeled_item_width(self.app._psim, "Radius", radius_width):
            radius_changed, scope_radius = self.app._psim.InputFloat("##scope_radius", self.app.settings.scope_radius)
        if radius_changed:
            self.app._set_settings(scope_radius=max(0.0, scope_radius))

    def _ui_marker_controls(self) -> None:  # pragma: no cover - Polyscope callback
        marker_length_width, marker_thickness_width, marker_max_width = ui_equal_widths(
            self.app._psim, 3, reserve=330.0, min_width=60.0, max_width=120.0
        )
        with ui_labeled_item_width(self.app._psim, "Marker Length", marker_length_width):
            marker_length_changed, marker_length = self.app._psim.InputFloat(
                "##marker_length", self.app.settings.scope_marker_length_px
            )
        if marker_length_changed:
            self.app._set_settings(scope_marker_length_px=max(0.0, marker_length))

        self.app._psim.SameLine()
        with ui_labeled_item_width(self.app._psim, "Marker Thickness", marker_thickness_width):
            marker_thickness_changed, marker_thickness = self.app._psim.InputFloat(
                "##marker_thickness", self.app.settings.scope_marker_thickness_px
            )
        if marker_thickness_changed:
            self.app._set_settings(scope_marker_thickness_px=max(0.0, marker_thickness))

        with ui_labeled_item_width(self.app._psim, "Marker Max", marker_max_width):
            marker_max_changed, marker_max = self.app._psim.InputFloat(
                "##marker_max", self.app.settings.scope_marker_max_fraction
            )
        if marker_max_changed:
            self.app._set_settings(scope_marker_max_fraction=max(0.0, marker_max))

    def _ui_messages(self) -> None:  # pragma: no cover - Polyscope callback
        for message in self.app._session.scope_msgs:
            self.app._psim.Text(message)

        for _ in range(max(0, _PADDING - len(self.app._session.scope_msgs))):
            self.app._psim.Text("")

    @freq_time_profiler("ui_scope")
    def ui(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self.app._psim, "Scope", open_first_time=False) as expanded:
            if not expanded:
                self.cleanup()
                return

            self._ui_primary_controls()
            self._ui_marker_controls()
            if self.app.settings.scope_enabled:
                self._handle_interaction()
                self._update_cross_geometry()

            self._ui_messages()
