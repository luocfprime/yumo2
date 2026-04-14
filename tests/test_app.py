# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

import numpy as np
import pytest

from yumo2.app import Config, PolyscopeApp
from yumo2.settings import Settings, load_profile


class _FakePolyscopeModule:
    def set_program_name(self, name: str) -> None:
        return None

    def set_print_prefix(self, prefix: str) -> None:
        return None

    def set_ground_plane_mode(self, mode: str) -> None:
        return None

    def set_up_dir(self, direction: str) -> None:
        return None

    def set_front_dir(self, direction: str) -> None:
        return None

    def init(self) -> None:
        return None

    def set_camera_view_matrix(self, matrix: np.ndarray) -> None:
        return None

    def get_camera_view_matrix(self) -> np.ndarray:
        return np.eye(4)

    def set_view_center_raw(self, center: np.ndarray) -> None:
        return None

    def set_user_callback(self, callback) -> None:
        return None

    def show(self) -> None:
        return None


def test_polyscope_app_initializes_without_loading_runtime_dependencies(tmp_path: Path) -> None:
    config = Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl")

    app = PolyscopeApp(config, root_dir=tmp_path)

    assert app.config == config
    assert app.active_profile == "default"
    assert app.settings.colormap == "RdBu"
    assert app.settings.material == "0.30_clay_0.70_flat"


def test_polyscope_app_uses_external_settings_without_loading_profile(tmp_path: Path) -> None:
    external = Settings(colormap="magma", snapshot_crop_h=123)

    app = PolyscopeApp(
        config=Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"),
        root_dir=tmp_path,
        settings=external,
    )

    assert app.active_profile == "external"
    assert app.settings is external
    assert app.settings.snapshot_crop_h == 123


def test_run_executes_stages_in_order(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class StagedApp(PolyscopeApp):
        def _load_mesh_and_uv(self) -> None:
            calls.append("mesh")

        def _load_data(self) -> None:
            calls.append("data")

        def _init_polyscope(self) -> None:
            calls.append("init")

        def _setup_scene(self) -> None:
            calls.append("scene")

    class FakePsModule(_FakePolyscopeModule):
        def set_user_callback(self, callback) -> None:
            calls.append("callback")

        def show(self) -> None:
            calls.append("show")

    import sys
    import types

    fake_ps = types.ModuleType("polyscope")
    fake_ps_module = FakePsModule()
    for name in dir(fake_ps_module):
        if not name.startswith("_"):
            setattr(fake_ps, name, getattr(fake_ps_module, name))

    fake_psim = types.ModuleType("polyscope.imgui")
    fake_ps.imgui = fake_psim

    monkeypatch.setitem(sys.modules, "polyscope", fake_ps)
    monkeypatch.setitem(sys.modules, "polyscope.imgui", fake_psim)

    app = StagedApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.run()

    assert calls == ["mesh", "data", "init", "scene", "callback", "show"]


def test_save_as_new_profile_creates_and_activates_profile(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"denoise_sigma": 2.0})
    app._session.save_as_new_enabled = True
    app._session.new_profile_name = "experiment"

    app._save_as_new_profile()

    assert app.active_profile == "experiment"
    assert load_profile("experiment", tmp_path).denoise_sigma == 2.0


def test_save_as_new_profile_rejects_blank_name(tmp_path: Path) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._session.save_as_new_enabled = True
    app._session.new_profile_name = "   "

    app._save_as_new_profile()

    assert app.active_profile == "default"
    assert "required" in app._session.profile_status_msg.lower()


@pytest.mark.parametrize("preexisting", [False, True])
def test_ensure_default_imgui_ini_handles_missing_and_existing_file(
    tmp_path: Path, monkeypatch, preexisting: bool
) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    target = run_dir / "imgui.ini"
    if preexisting:
        target.write_text("custom imgui config\n")

    app._ensure_default_imgui_ini()

    assert target.exists()
    if preexisting:
        assert target.read_text() == "custom imgui config\n"
    else:
        assert target.read_text() == app._default_imgui_ini_path().read_text()


def test_load_materials_skips_already_loaded_names(tmp_path: Path, monkeypatch) -> None:
    import yumo2.app as app_module

    monkeypatch.setattr(app_module, "_LOADED_MATERIALS", set())

    for suffix in ("0", "1"):
        (tmp_path / f"ceramic_{suffix}.hdr").write_bytes(b"hdr")

    class FakePs(_FakePolyscopeModule):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def load_blendable_material(self, name: str, filename_base: str, filename_ext: str) -> None:
            self.calls.append((name, filename_base, filename_ext))

    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app._ps = FakePs()

    first = app._load_materials(tmp_path)
    second = app._load_materials(tmp_path)

    assert first[0] == "ceramic"
    assert second[0] == "ceramic"
    assert app._ps.calls == [("ceramic", str(tmp_path / "ceramic"), ".hdr")]


def test_rebuild_texture_applies_log_transform_after_sampling(tmp_path: Path, monkeypatch) -> None:
    app = PolyscopeApp(Config(data_path=tmp_path / "field.plt", mesh_path=tmp_path / "mesh.stl"), root_dir=tmp_path)
    app.settings = app.settings.model_copy(update={"scalar_transform": "log_10", "denoise_enabled": False})

    app._session.scalar_field = np.array([
        [[1.0, 1.0], [1.0, 1.0]],
        [[1000.0, 1000.0], [1000.0, 1000.0]],
    ])
    app._session.xs = np.array([0.0, 1.0])
    app._session.ys = np.array([0.0, 1.0])
    app._session.zs = np.array([0.0, 1.0])
    app._session.uvs = np.zeros((3, 2), dtype=float)
    app._session.faces_unwrapped = np.zeros((1, 3), dtype=np.int32)
    app._session.vertices_unwrapped = np.zeros((3, 3), dtype=float)
    app._session.texture_height = 1
    app._session.texture_width = 1

    monkeypatch.setattr(
        "yumo2.app.sample_texture_positions",
        lambda vertices_unwrapped, faces_unwrapped, uvs, height, width: (
            np.array([[0]], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([[0.5, 0.5, 0.5]], dtype=float),
        ),
    )
    app._refresh_quantities = lambda: None

    app._rebuild_texture()

    assert app._session.original_texture is not None
    assert app._session.raw_texture is not None
    assert app._session.effective_range is not None
    assert np.isclose(app._session.original_texture[0, 0], 500.5)
    assert np.isclose(app._session.raw_texture[0, 0], np.log10(500.5))
    assert app._session.effective_range == (np.log10(500.5), np.log10(500.5))
    assert not np.isclose(app._session.raw_texture[0, 0], 1.5)
