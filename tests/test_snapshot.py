# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from contextlib import contextmanager
from pathlib import Path

import numpy as np

import yumo2.features.snapshot as snapshot_module
from yumo2.app import Config, PolyscopeApp


@contextmanager
def _expanded_tree(*args, **kwargs):
    yield True


class _SnapshotUiPsim:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def GetContentRegionAvail(self):
        return (360.0, 200.0)

    def AlignTextToFramePadding(self) -> None:
        return None

    def PushItemWidth(self, width: float) -> None:
        self.calls.append(("PushItemWidth", width))

    def PopItemWidth(self) -> None:
        self.calls.append(("PopItemWidth",))

    def InputText(self, label: str, value: str):
        self.calls.append(("InputText", label, value))
        return False, value

    def Checkbox(self, label: str, value: bool):
        self.calls.append(("Checkbox", label, value))
        return False, value

    def SameLine(self) -> None:
        self.calls.append(("SameLine",))

    def Button(self, label: str) -> bool:
        self.calls.append(("Button", label))
        return label == "Refresh"

    def Text(self, value: str) -> None:
        self.calls.append(("Text", value))


def test_snap_make_colorbar_uses_rgba_size_controls(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._session.effective_range = (0.0, 1.0)
    app.settings = app.settings.model_copy(update={"snapshot_colorbar_h": 60, "snapshot_colorbar_w": 150})

    colorbar = app.snapshot.make_colorbar()

    assert colorbar.shape[2] == 4
    assert colorbar.shape[1] == 150
    assert colorbar.shape[0] >= 60


def test_snap_preview_rgb_overlays_colorbar_with_xy_offset(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "snapshot_crop_h": 40,
            "snapshot_crop_w": 50,
            "snapshot_colorbar_x": 5,
            "snapshot_colorbar_y": 7,
        }
    )

    rgba = np.zeros((80, 100, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = 255

    app.snapshot.make_colorbar = lambda: np.dstack([
        np.full((10, 12), 1.0),
        np.zeros((10, 12)),
        np.zeros((10, 12)),
        np.ones((10, 12)),
    ])

    composed = app.snapshot.preview_rgb(rgba)

    assert np.allclose(composed[7, 5], [1.0, 0.0, 0.0])
    assert np.allclose(composed[0, 0], [1.0, 1.0, 1.0])


def test_snap_composed_rgba_preserves_transparent_background(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "snapshot_crop_h": 20,
            "snapshot_crop_w": 20,
            "snapshot_colorbar_x": 5,
            "snapshot_colorbar_y": 5,
        }
    )

    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    app.snapshot.make_colorbar = lambda: np.dstack([
        np.full((4, 4), 1.0),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.ones((4, 4)),
    ])

    composed = app.snapshot.composed_rgba(rgba)

    assert composed.shape[2] == 4
    assert np.allclose(composed[0, 0], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(composed[5, 5], [1.0, 0.0, 0.0, 1.0])


def test_snap_ui_throttles_live_preview_by_time_not_frame_count(tmp_path: Path, monkeypatch) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"snapshot_live_preview": True})

    class _FakeIO:
        Framerate = 2.0

    class _FakePsim:
        @staticmethod
        def GetIO():
            return _FakeIO()

    @contextmanager
    def _collapsed_tree(*args, **kwargs):
        yield False

    times = iter([0.0, 0.05, 0.2])
    update_calls: list[int] = []

    monkeypatch.setattr(snapshot_module, "ui_tree_node", _collapsed_tree)
    app._psim = _FakePsim()
    monkeypatch.setattr(snapshot_module.time, "monotonic", lambda: next(times))
    app.snapshot.capture_rgba = lambda: np.zeros((8, 8, 4), dtype=np.uint8)
    app.snapshot.update_textures = lambda rgba: update_calls.append(1)
    app.snapshot.draw_windows = lambda: None

    app.snapshot.ui()
    app.snapshot.ui()
    app.snapshot.ui()

    assert len(update_calls) == 2


def test_snap_ui_exposes_manual_refresh_button(tmp_path: Path, monkeypatch) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"snapshot_live_preview": False})
    app._session.snapshot_textures_initialized = True

    refresh_calls: list[int] = []
    monkeypatch.setattr(snapshot_module, "ui_tree_node", _expanded_tree)
    app._psim = _SnapshotUiPsim()
    app.snapshot.capture_rgba = lambda: np.zeros((8, 8, 4), dtype=np.uint8)
    app.snapshot.update_textures = lambda rgba: refresh_calls.append(1)
    app.snapshot.draw_windows = lambda: None
    app.snapshot._ui_image_controls = lambda width: None
    app.snapshot._ui_colorbar_controls = lambda width: None

    app.snapshot.ui()

    assert refresh_calls == [1]
    assert ("Text", "Filename") in app._psim.calls
    assert ("InputText", "##snap", app._session.snapshot_filename) in app._psim.calls
    assert ("Checkbox", "##snapshot_live_preview", app.settings.snapshot_live_preview) in app._psim.calls
    assert ("Button", "Refresh") in app._psim.calls
    assert ("Button", "Save PNG") in app._psim.calls
