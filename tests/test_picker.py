# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

import numpy as np

from yumo2.app import Config, PolyscopeApp


def test_mesh_pick_values_returns_surface_and_smoothed(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._session.uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    app._session.faces_unwrapped = np.array([[0, 1, 2]])
    app._session.original_texture = np.array([
        [100.0, 200.0],
        [300.0, 400.0],
    ])
    app._session.raw_texture = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    app._session.texture = np.array([
        [11.0, 21.0],
        [31.0, 41.0],
    ])

    surface_value, smoothed_value, original_value = app.picker.mesh_pick_values(0, np.array([0.0, 0.0, 1.0]))

    assert surface_value == 10.0
    assert smoothed_value == 11.0
    assert original_value == 100.0


def test_picker_ui_formats_world_position_to_three_decimals(tmp_path: Path, monkeypatch) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)

    class _FakeIO:
        def __init__(self) -> None:
            self.MouseClicked = [True]
            self.MousePos = (100.0, 50.0)

    class _FakePsim:
        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def GetIO(self):
            return _FakeIO()

        def Text(self, message: str) -> None:
            self.text_calls.append(message)

    class _FakePickResult:
        def __init__(self) -> None:
            self.is_hit = False
            self.structure_name = ""
            self.structure_data: dict[str, object] = {}

    class _FakePs:
        @staticmethod
        def screen_coords_to_world_position(screen_coords):
            return np.array([1.23456, 2.34567, 3.45678])

        @staticmethod
        def pick(screen_coords):
            return _FakePickResult()

    from contextlib import contextmanager

    @contextmanager
    def _expanded_tree(*args, **kwargs):
        yield True

    monkeypatch.setattr("yumo2.features.picker.ui_tree_node", _expanded_tree)
    app._psim = _FakePsim()
    app._ps = _FakePs()

    app.picker.ui()

    assert app._session.picker_msgs[0] == "World coord: [1.235, 2.346, 3.457]"
    assert app._psim.text_calls[0] == "World coord: [1.235, 2.346, 3.457]"


def test_picker_ui_shows_original_value_for_log_transform(tmp_path: Path, monkeypatch) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"scalar_transform": "log_10"})

    class _FakeIO:
        def __init__(self) -> None:
            self.MouseClicked = [True]
            self.MousePos = (100.0, 50.0)

    class _FakePsim:
        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def GetIO(self):
            return _FakeIO()

        def Text(self, message: str) -> None:
            self.text_calls.append(message)

    class _FakePickResult:
        def __init__(self) -> None:
            self.is_hit = True
            self.structure_name = "mesh"
            self.structure_data: dict[str, object] = {"index": 0, "bary_coords": [0.0, 0.0, 1.0]}

    class _FakePs:
        @staticmethod
        def screen_coords_to_world_position(screen_coords):
            return np.array([1.0, 2.0, 3.0])

        @staticmethod
        def pick(screen_coords):
            return _FakePickResult()

    from contextlib import contextmanager

    @contextmanager
    def _expanded_tree(*args, **kwargs):
        yield True

    monkeypatch.setattr("yumo2.features.picker.ui_tree_node", _expanded_tree)
    app._psim = _FakePsim()
    app._ps = _FakePs()
    app.picker.mesh_pick_values = lambda face_index, barycentric_coords: (2.0, None, 1234.567)

    app.picker.ui()

    assert "Surface value: 2" in app._session.picker_msgs
    assert "Original surface value: 1.234567e+03" in app._session.picker_msgs
