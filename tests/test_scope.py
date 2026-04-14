# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

import numpy as np

from yumo2.app import Config, PolyscopeApp
from yumo2.features.scope import ScopeQueryResult


class _FakeCurveNetwork:
    def __init__(self, nodes: np.ndarray) -> None:
        self.nodes = nodes.copy()
        self.enabled = True
        self.radius = None

    def set_color(self, color) -> None:
        return None

    def set_material(self, material: str) -> None:
        return None

    def set_radius(self, radius: float, relative: bool) -> None:
        self.radius = (radius, relative)

    def update_node_positions(self, nodes: np.ndarray) -> None:
        self.nodes = nodes.copy()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


def test_scope_show_cross_uses_screen_proportional_marker(tmp_path: Path) -> None:
    import math

    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "scope_marker_length_px": 30.0,
            "scope_marker_thickness_px": 3.0,
            "scope_marker_max_fraction": 1.0,
        }
    )
    app._session.vertices = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    class _FakeCam:
        def get_position(self):
            return (0.0, 0.0, 10.0)  # camera 10 units away from origin

        def get_fov_vertical_deg(self):
            return 60.0

    class _FakePsim:
        class _FakeIO:
            DisplaySize = (1280.0, 720.0)

        def GetIO(self):
            return self._FakeIO()

    class FakePs:
        def __init__(self) -> None:
            self.curve = None

        def get_view_camera_parameters(self):
            return _FakeCam()

        def register_curve_network(self, name: str, nodes: np.ndarray, edges: np.ndarray, enabled: bool = True):
            self.curve = _FakeCurveNetwork(nodes)
            self.curve.enabled = enabled
            return self.curve

    app._ps = FakePs()
    app._psim = _FakePsim()

    app.scope._show_cross(np.zeros(3))

    # dist=10, fov=60°, screen_h=720
    dist, fov_rad, screen_h = 10.0, math.radians(60.0), 720.0
    world_per_pixel = (2.0 * dist * math.tan(fov_rad / 2.0)) / screen_h

    expected_length = app.settings.scope_marker_length_px * world_per_pixel
    expected_radius = 0.5 * app.settings.scope_marker_thickness_px * world_per_pixel
    assert np.allclose(app._ps.curve.nodes[0], [-expected_length / 2.0, 0.0, 0.0])
    assert np.allclose(app._ps.curve.nodes[1], [expected_length / 2.0, 0.0, 0.0])
    assert app._ps.curve.radius == (expected_radius, False)


def test_scope_show_cross_scales_world_size_with_camera_distance(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "scope_marker_length_px": 30.0,
            "scope_marker_thickness_px": 3.0,
            "scope_marker_max_fraction": 1.0,
        }
    )
    app._session.vertices = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    class _FakeCam:
        def __init__(self, z: float) -> None:
            self.z = z

        def get_position(self):
            return (0.0, 0.0, self.z)

        def get_fov_vertical_deg(self):
            return 60.0

    class _FakePsim:
        class _FakeIO:
            DisplaySize = (1280.0, 720.0)

        def GetIO(self):
            return self._FakeIO()

    class FakePs:
        def __init__(self) -> None:
            self.curve = None
            self.cam = _FakeCam(10.0)

        def get_view_camera_parameters(self):
            return self.cam

        def register_curve_network(self, name: str, nodes: np.ndarray, edges: np.ndarray, enabled: bool = True):
            self.curve = _FakeCurveNetwork(nodes)
            self.curve.enabled = enabled
            return self.curve

    app._ps = FakePs()
    app._psim = _FakePsim()

    app.scope._show_cross(np.zeros(3))
    near_nodes = app._ps.curve.nodes.copy()
    near_radius = app._ps.curve.radius[0]

    app._ps.cam = _FakeCam(20.0)
    app.scope._update_cross_geometry()

    far_nodes = app._ps.curve.nodes
    far_radius = app._ps.curve.radius[0]

    near_length = near_nodes[1, 0] - near_nodes[0, 0]
    far_length = far_nodes[1, 0] - far_nodes[0, 0]

    assert np.isclose(far_length / near_length, 2.0)
    assert np.isclose(far_radius / near_radius, 2.0)


def test_scope_show_cross_caps_marker_size_by_scene_fraction(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "scope_marker_length_px": 30.0,
            "scope_marker_thickness_px": 3.0,
            "scope_marker_max_fraction": 0.02,
        }
    )
    app._session.vertices = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ])

    class _FakeCam:
        def get_position(self):
            return (0.0, 0.0, 200.0)

        def get_fov_vertical_deg(self):
            return 60.0

    class _FakePsim:
        class _FakeIO:
            DisplaySize = (1280.0, 720.0)

        def GetIO(self):
            return self._FakeIO()

    class FakePs:
        def __init__(self) -> None:
            self.curve = None

        def get_view_camera_parameters(self):
            return _FakeCam()

        def register_curve_network(self, name: str, nodes: np.ndarray, edges: np.ndarray, enabled: bool = True):
            self.curve = _FakeCurveNetwork(nodes)
            self.curve.enabled = enabled
            return self.curve

    app._ps = FakePs()
    app._psim = _FakePsim()

    app.scope._show_cross(np.zeros(3))

    cross_length = app._ps.curve.nodes[1, 0] - app._ps.curve.nodes[0, 0]
    assert np.isclose(cross_length, 2.0)
    assert app._ps.curve.radius == (0.16, False)


def test_scope_extremum_uses_raw_values_for_selection(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._session.original_texture = np.array([[100.0, 200.0]])
    app._session.raw_texture = np.array([[1.0, 2.0]])
    app._session.texture = np.array([[10.0, 0.5]])
    app._session.texel_rows = np.array([0, 0], dtype=np.int64)
    app._session.texel_cols = np.array([0, 1], dtype=np.int64)
    app._session.texel_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])

    result = app.scope.query_extremum(center=np.array([0.0, 0.0, 0.0]), radius=1.0, mode="Max")

    assert result is not None
    assert np.allclose(result.position, [0.5, 0.0, 0.0])
    assert result.raw_value == 2.0
    assert result.smoothed_value == 0.5
    assert result.original_value == 200.0


def test_scope_handle_interaction_formats_positions_to_three_decimals(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._session.scope_press_world = np.array([1.23456, 2.34567, 3.45678])

    class _FakeIO:
        def __init__(self) -> None:
            self.MouseClicked = [False]
            self.MouseDown = [False]
            self.MouseReleased = [True]
            self.MousePos = (0.0, 0.0)
            self.DisplaySize = (1280.0, 720.0)

    class _FakePsim:
        @staticmethod
        def GetIO():
            return _FakeIO()

    app._psim = _FakePsim()
    app.scope._hide_sphere = lambda: None
    app.scope._hide_cross = lambda: None
    app.scope._show_cross = lambda center: None
    app.scope.query_extremum = lambda center, radius, mode: ScopeQueryResult(
        position=np.array([9.87654, 8.76543, 7.65432]),
        raw_value=12.0,
        smoothed_value=None,
        original_value=12.0,
    )

    app.scope._handle_interaction()

    assert app._session.scope_msgs[0] == "Scope center: [1.235, 2.346, 3.457]"
    assert app._session.scope_msgs[1] == "Scope max position: [9.877, 8.765, 7.654]"


def test_scope_handle_interaction_shows_original_value_for_log_transform(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"scalar_transform": "log_10"})
    app._session.scope_press_world = np.array([1.0, 2.0, 3.0])

    class _FakeIO:
        def __init__(self) -> None:
            self.MouseClicked = [False]
            self.MouseDown = [False]
            self.MouseReleased = [True]
            self.MousePos = (0.0, 0.0)
            self.DisplaySize = (1280.0, 720.0)

    class _FakePsim:
        @staticmethod
        def GetIO():
            return _FakeIO()

    app._psim = _FakePsim()
    app.scope._hide_sphere = lambda: None
    app.scope._hide_cross = lambda: None
    app.scope._show_cross = lambda center: None
    app.scope.query_extremum = lambda center, radius, mode: ScopeQueryResult(
        position=np.array([9.0, 8.0, 7.0]),
        raw_value=2.0,
        smoothed_value=None,
        original_value=1234.567,
    )

    app.scope._handle_interaction()

    assert "Raw surface value: 2" in app._session.scope_msgs
    assert "Original surface value: 1.234567e+03" in app._session.scope_msgs
