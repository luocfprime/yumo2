# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from yumo2.__about__ import __version__
from yumo2.colormap import load_colormaps
from yumo2.constants import (
    ASSETS_ROOT,
    BUILTIN_COLORMAPS,
    BUILTIN_MATERIALS,
    DEFAULT_COLORMAP,
    DEFAULT_IMGUI_INI,
    DEFAULT_MATERIAL,
    FONT_PATH,
)
from yumo2.features import Picker, Scope, Snapshot
from yumo2.loader import load_mesh, load_scalar_field
from yumo2.profiling import freq_time_profiler, mem_profiler, time_profiler
from yumo2.settings import (
    Settings,
    list_profiles,
    load_active_profile,
    load_profile,
    save_profile,
    set_active_profile,
)
from yumo2.texture import apply_denoise, sample_texture_positions, trilinear_sample, unwrap_uv
from yumo2.ui import (
    ui_available_width,
    ui_equal_widths,
    ui_labeled_checkbox,
    ui_labeled_item_width,
    ui_select,
    ui_tree_node,
)
from yumo2.utils import finite_minmax, transform_scalar_data

logger = structlog.get_logger(__name__)

_LOADED_MATERIALS: set[str] = set()


@dataclass(frozen=True)
class Config:
    data_path: Path
    mesh_path: Path


@dataclass
class SessionContext:
    """Transient state for one session (one loaded dataset + settings pair).
    Everything here is ephemeral — nothing is persisted outside of Settings.

    Populated in stages:
      _reset_session_context() → settings-derived fields (saved_camera_view, etc.)
      _load_data()             → scalar_field, xs, ys, zs
      _load_mesh_and_uv()      → vertices, faces, UV fields, texture dimensions
      _rebuild_texture()       → uv_mask, original_texture, raw_texture, texture, effective_range
    """

    # --- scalar field ---
    scalar_field: np.ndarray | None = None
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None
    zs: np.ndarray | None = None

    # --- mesh + UV ---
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    vertices_unwrapped: np.ndarray | None = None
    faces_unwrapped: np.ndarray | None = None
    uvs: np.ndarray | None = None
    param_corner: np.ndarray | None = None
    texture_height: int | None = None
    texture_width: int | None = None

    # --- baked texture ---
    uv_mask: np.ndarray | None = None
    original_texture: np.ndarray | None = None
    raw_texture: np.ndarray | None = None
    texture: np.ndarray | None = None
    texel_rows: np.ndarray | None = None
    texel_cols: np.ndarray | None = None
    texel_positions: np.ndarray | None = None

    # --- settings-derived / UI state ---
    saved_camera_view: np.ndarray | None = None
    effective_range: tuple[float, float] | None = None
    setting_unsaved: bool = False
    save_as_new_enabled: bool = False
    new_profile_name: str = ""
    profile_status_msg: str = ""
    picker_msgs: list[str] = field(default_factory=list)
    scope_msgs: list[str] = field(default_factory=list)
    scope_press_world: np.ndarray | None = None
    scope_cross_center: np.ndarray | None = None
    snapshot_status_msg: str = ""
    snapshot_filename: str = ""
    snapshot_textures_initialized: bool = False
    snapshot_last_preview_time: float | None = None


class PolyscopeApp:
    def __init__(self, config: Config, root_dir: Path | None = None, settings: Settings | None = None):
        self.config = config
        self.root_dir = root_dir or Path.cwd()

        initial_profile: str
        initial_settings: Settings
        if settings is not None:
            initial_profile = "external"
            initial_settings = settings
        else:
            initial_profile, initial_settings = load_active_profile(self.root_dir)

        self._ps: Any = None
        self._psim: Any = None
        self._mesh: Any = None
        self._loaded_cmaps: dict[str, str] = {}
        self._available_colormaps = list(BUILTIN_COLORMAPS)
        self._available_materials = list(BUILTIN_MATERIALS)

        self.active_profile: str
        self.settings: Settings
        self._session: SessionContext
        self.snapshot = Snapshot(self)
        self.scope = Scope(self)
        self.picker = Picker(self)

        self._reset_session_context(initial_settings, active_profile=initial_profile)

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self) -> None:
        """Launch the GUI and drive the full application lifecycle."""
        import polyscope as ps
        import polyscope.imgui as psim

        self._ensure_default_imgui_ini()
        self._ps = ps
        self._psim = psim

        self.pre_load_hook()
        self._load_mesh_and_uv()
        self._load_data()
        self.post_load_hook()

        self._init_polyscope()
        self.post_polyscope_init_hook()

        self._setup_scene()
        self.post_scene_setup_hook()

        ps.set_user_callback(self.callback)

        try:
            ps.show()
        finally:
            self.scope.cleanup()
            self.snapshot.cleanup()

    @property
    def fps(self) -> float:
        if self._psim is None:
            return 0.0
        return float(self._psim.GetIO().Framerate)

    @freq_time_profiler("callback")
    def callback(self) -> None:  # pragma: no cover - Polyscope callback
        self._ui_info_section()
        self._ui_profile_section()
        self._ui_appearance_section()
        self._ui_processing_section()
        self.snapshot.ui()
        self.scope.ui()
        self.picker.ui()

    # ── Extension Hooks ──────────────────────────────────────────────────────
    def pre_load_hook(self) -> None:
        """Public extension point before any data or Polyscope state is loaded."""

    def post_load_hook(self) -> None:
        """Public extension point after mesh/data load, before Polyscope init."""

    def post_polyscope_init_hook(self) -> None:
        """Public extension point after Polyscope init, before scene setup."""

    def post_scene_setup_hook(self) -> None:
        """Public extension point after scene setup, before entering the UI loop."""

    # ── Internal Helpers ─────────────────────────────────────────────────────
    def _default_profile_name(self) -> str:
        return datetime.now().strftime("%y%m%d-%H%M%S")

    def _default_imgui_ini_path(self) -> Path:
        return DEFAULT_IMGUI_INI

    def _ensure_default_imgui_ini(self) -> None:
        target = Path.cwd() / "imgui.ini"
        if target.exists():
            return
        shutil.copyfile(self._default_imgui_ini_path(), target)
        logger.debug("default_imgui_ini_copied", source=str(self._default_imgui_ini_path()), target=str(target))

    def _log_settings_changed(self, updates: dict[str, object]) -> None:
        changes = {
            key: {"from": getattr(self.settings, key), "to": value}
            for key, value in updates.items()
            if getattr(self.settings, key) != value
        }
        if changes:
            logger.debug("settings_changed", changes=changes)

    # ── Pipeline stages ──────────────────────────────────────────────────────

    @time_profiler("load_mesh_and_uv")
    @mem_profiler("load_mesh_and_uv")
    def _load_mesh_and_uv(self) -> None:
        """Load mesh and run UV unwrap. Slow (xatlas). Only needs to run once per mesh."""
        logger.info("loading_mesh", path=str(self.config.mesh_path))
        vertices, faces = load_mesh(self.config.mesh_path)
        self._session.vertices, self._session.faces = vertices, faces
        logger.debug("mesh_loaded", vertices=len(vertices), faces=len(faces))

        logger.debug("unwrapping_uv")
        (
            self._session.uvs,
            self._session.faces_unwrapped,
            self._session.vertices_unwrapped,
            self._session.param_corner,
            atlas_height,
            atlas_width,
        ) = unwrap_uv(vertices, faces)
        self._session.texture_height = atlas_height
        self._session.texture_width = atlas_width
        logger.debug("uv_unwrapped", atlas_height=atlas_height, atlas_width=atlas_width)

    @time_profiler("load_data")
    @mem_profiler("load_data")
    def _load_data(self) -> None:
        """Load scalar field from config.data_path. Fast; re-call when data changes."""
        logger.info("loading_data", path=str(self.config.data_path))
        self._session.scalar_field, self._session.xs, self._session.ys, self._session.zs = load_scalar_field(
            self.config.data_path
        )
        logger.debug("data_loaded", shape=self._session.scalar_field.shape)

    def _init_polyscope(self) -> None:
        """Initialize Polyscope and load assets. Call once per process."""
        logger.debug("initializing_polyscope")
        ps = self._ps
        ps.set_program_name("yumo2")
        ps.set_print_prefix("[Yumo2][Polyscope] ")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("z_up")
        ps.set_front_dir("x_front")
        if FONT_PATH.exists():

            def _prepare_fonts(font_atlas):
                font = font_atlas.AddFontFromFileTTF(str(FONT_PATH), 20.0)
                return font, font

            ps.set_prepare_imgui_fonts_callback(_prepare_fonts)
        else:
            logger.warning("cjk_font_not_found", path=str(FONT_PATH))
        ps.init()

        self._loaded_cmaps = load_colormaps(ps, ASSETS_ROOT / "colormaps")
        self._available_colormaps = [*self._loaded_cmaps.keys(), *BUILTIN_COLORMAPS]
        self._available_materials = self._load_materials(ASSETS_ROOT / "materials")
        logger.debug(
            "polyscope_assets_loaded",
            custom_colormaps=list(self._loaded_cmaps.keys()),
            materials=self._available_materials,
        )

    @time_profiler("setup_scene")
    @mem_profiler("setup_scene")
    def _setup_scene(self) -> None:
        """Register mesh, bake texture, and initialize camera. Re-callable per render."""
        logger.debug("setting_up_scene", profile=self.active_profile)
        self._normalize_default_settings()
        self._register_mesh()
        self._rebuild_texture()
        if self._session.saved_camera_view is not None:
            self._ps.set_camera_view_matrix(self._session.saved_camera_view.copy())
            logger.debug("camera_view_restored")
        else:
            # No saved view — set orbit center to mesh bbox, then capture as reset target.
            self._apply_scene_view_center()
            self._session.saved_camera_view = np.array(self._ps.get_camera_view_matrix(), dtype=float)
            logger.debug("camera_view_initialised_from_bbox")

    def _reset_session_context(self, settings: Settings, active_profile: str | None = None) -> None:
        """Reset all session-scoped state for a new session (new dataset + settings pair).
        Session context is transient — nothing here is persisted independently of Settings."""
        self.settings = settings
        if active_profile is not None:
            self.active_profile = active_profile
        self.scope.cleanup()
        self._session = SessionContext(
            saved_camera_view=(
                np.array(settings.camera_view, dtype=float) if settings.camera_view is not None else None
            ),
            new_profile_name=self._default_profile_name(),
            snapshot_filename=self.snapshot.default_filename(),
        )

    def _register_mesh(self) -> None:
        if self._ps is None:
            return

        self._mesh = self._ps.register_surface_mesh(
            "mesh", self._session.vertices_unwrapped, self._session.faces_unwrapped
        )
        self._mesh.set_color((0.75, 0.75, 0.75))
        self._mesh.set_material(self.settings.material)
        self._mesh.add_parameterization_quantity("uv", self._session.param_corner, defined_on="corners", enabled=True)

    def _scene_center(self) -> np.ndarray:
        if self._session.vertices is None or len(self._session.vertices) == 0:
            return np.zeros(3, dtype=float)
        bbox_min = np.min(self._session.vertices, axis=0)
        bbox_max = np.max(self._session.vertices, axis=0)
        return (bbox_min + bbox_max) / 2.0

    def _apply_scene_view_center(self) -> None:
        if self._ps is None:
            return
        self._ps.set_view_center_raw(self._scene_center())

    def _normalize_default_settings(self) -> None:
        if self.settings.colormap == DEFAULT_COLORMAP and self._loaded_cmaps:
            self.settings = self.settings.model_copy(update={"colormap": next(iter(self._loaded_cmaps))})

        if self.settings.material == DEFAULT_MATERIAL and DEFAULT_MATERIAL not in self._available_materials:
            fallback_material = self._available_materials[0] if self._available_materials else DEFAULT_MATERIAL
            self.settings = self.settings.model_copy(update={"material": fallback_material})

    def _load_materials(self, *directories: Path | None) -> list[str]:
        if self._ps is None:
            return list(BUILTIN_MATERIALS)

        discovered: list[str] = []
        for directory in directories:
            if directory is None or not directory.exists():
                continue
            for stem in sorted({path.stem.rsplit("_", 1)[0] for path in directory.glob("*_?.hdr")}):
                if stem in _LOADED_MATERIALS:
                    logger.debug("material_already_loaded", material=stem, directory=str(directory))
                    discovered.append(stem)
                    continue
                try:
                    self._ps.load_blendable_material(
                        stem,
                        filename_base=str(directory / stem),
                        filename_ext=".hdr",
                    )
                    discovered.append(stem)
                    _LOADED_MATERIALS.add(stem)
                except Exception as exc:  # pragma: no cover - depends on Polyscope runtime
                    logger.warning("failed_to_load_material", material=stem, directory=str(directory), error=str(exc))

        return [*discovered, *BUILTIN_MATERIALS]

    def _reset_camera_view(self) -> None:
        if self._ps is None or self._session.saved_camera_view is None:
            return
        self._ps.set_camera_view_matrix(self._session.saved_camera_view.copy())

    def _save_active_profile(self) -> None:
        if self._ps is not None:
            current_camera = np.array(self._ps.get_camera_view_matrix(), dtype=float)
            self._session.saved_camera_view = current_camera.copy()
            self.settings = self.settings.model_copy(update={"camera_view": current_camera.tolist()})
        save_profile(self.active_profile, self.settings, self.root_dir)
        self._session.setting_unsaved = False
        self._session.profile_status_msg = f"Saved profile '{self.active_profile}'"

    def _save_as_new_profile(self) -> None:
        profile_name = self._session.new_profile_name.strip()
        if not profile_name:
            self._session.profile_status_msg = "Profile name is required"
            return

        self.active_profile = profile_name
        self._save_active_profile()
        self._session.save_as_new_enabled = False
        self._session.new_profile_name = self._default_profile_name()

    def _current_range(self) -> tuple[float, float]:
        rows = self._session.texel_rows
        cols = self._session.texel_cols
        original_texture = self._session.original_texture
        if original_texture is not None and rows is not None and cols is not None and len(rows) > 0:
            transformed_samples = transform_scalar_data(original_texture[rows, cols], self.settings.scalar_transform)
            return finite_minmax(transformed_samples)

        scalar_field = self._session.scalar_field
        if scalar_field is None:
            raise RuntimeError("Scalar field not loaded")
        transformed_field = transform_scalar_data(scalar_field, self.settings.scalar_transform)
        return finite_minmax(transformed_field)

    def _effective_vminmax(self) -> tuple[float, float]:
        if self._session.effective_range is None:
            self._session.effective_range = self._current_range()

        color_min = self.settings.color_min if self.settings.color_min is not None else self._session.effective_range[0]
        color_max = self.settings.color_max if self.settings.color_max is not None else self._session.effective_range[1]
        return color_min, color_max

    @time_profiler("rebuild_texture")
    @mem_profiler("rebuild_texture")
    def _rebuild_texture(self) -> None:
        scalar_field = self._session.scalar_field
        uvs = self._session.uvs
        faces_unwrapped = self._session.faces_unwrapped
        vertices_unwrapped = self._session.vertices_unwrapped
        xs = self._session.xs
        ys = self._session.ys
        zs = self._session.zs
        texture_height = self._session.texture_height
        texture_width = self._session.texture_width
        if scalar_field is None:
            raise RuntimeError("Texture rebuild requires loaded scalar field")
        if uvs is None or faces_unwrapped is None or vertices_unwrapped is None:
            raise RuntimeError("Texture rebuild requires loaded mesh UV state")
        if xs is None or ys is None or zs is None:
            raise RuntimeError("Texture rebuild requires loaded meshgrid axes")
        if texture_height is None or texture_width is None:
            raise RuntimeError("Texture rebuild requires texture dimensions")

        logger.debug(
            "rebuilding_texture",
            height=texture_height,
            width=texture_width,
            scalar_transform=self.settings.scalar_transform,
            denoise=self.settings.denoise_enabled,
            sigma=self.settings.denoise_sigma,
        )
        face_map, texel_rows, texel_cols, texel_positions = sample_texture_positions(
            vertices_unwrapped,
            faces_unwrapped,
            uvs,
            texture_height,
            texture_width,
        )
        self._session.uv_mask = (face_map >= 0).astype(np.float64)
        original_texture = np.zeros((texture_height, texture_width), dtype=np.float64)
        raw_texture = np.zeros((texture_height, texture_width), dtype=np.float64)
        transformed_samples = None
        if len(texel_rows) > 0:
            original_samples = trilinear_sample(texel_positions, scalar_field, xs, ys, zs)
            transformed_samples = transform_scalar_data(original_samples, self.settings.scalar_transform)
            original_texture[texel_rows, texel_cols] = original_samples
            raw_texture[texel_rows, texel_cols] = transformed_samples
        self._session.original_texture = original_texture
        self._session.raw_texture = raw_texture
        self._session.texel_rows = texel_rows
        self._session.texel_cols = texel_cols
        self._session.texel_positions = texel_positions

        if self.settings.denoise_enabled and self.settings.denoise_sigma > 0:
            self._session.texture = apply_denoise(raw_texture, self._session.uv_mask, self.settings.denoise_sigma)
        else:
            self._session.texture = raw_texture

        self._session.effective_range = (
            finite_minmax(transformed_samples) if transformed_samples is not None else self._current_range()
        )
        self._refresh_quantities()

    def _refresh_quantities(self) -> None:
        if self._mesh is None or self._session.texture is None:
            return

        self._mesh.set_material(self.settings.material)
        self._mesh.remove_quantity("texture", error_if_absent=False)
        self._mesh.add_scalar_quantity(
            "texture",
            self._session.texture,
            defined_on="texture",
            param_name="uv",
            image_origin="upper_left",
            cmap=self.settings.colormap,
            vminmax=self._effective_vminmax(),
            enabled=True,
        )

    def _set_settings(self, **updates: object) -> None:
        self._log_settings_changed(updates)
        self.settings = self.settings.model_copy(update=updates)
        self._session.setting_unsaved = True

    def _switch_profile(self, next_profile: str) -> None:
        logger.info("switching_profile", profile=next_profile)
        set_active_profile(next_profile, self.root_dir)
        old = self._session
        self._reset_session_context(load_profile(next_profile, self.root_dir), active_profile=next_profile)
        # Preserve loaded data — only settings change when switching profiles.
        self._session.scalar_field = old.scalar_field
        self._session.xs = old.xs
        self._session.ys = old.ys
        self._session.zs = old.zs
        self._session.vertices = old.vertices
        self._session.faces = old.faces
        self._session.vertices_unwrapped = old.vertices_unwrapped
        self._session.faces_unwrapped = old.faces_unwrapped
        self._session.uvs = old.uvs
        self._session.param_corner = old.param_corner
        self._session.texture_height = old.texture_height
        self._session.texture_width = old.texture_width
        self._session.texel_rows = old.texel_rows
        self._session.texel_cols = old.texel_cols
        self._session.texel_positions = old.texel_positions
        self._rebuild_texture()
        if self._session.saved_camera_view is not None:
            self._ps.set_camera_view_matrix(self._session.saved_camera_view.copy())
        elif self._ps is not None:
            self._session.saved_camera_view = np.array(self._ps.get_camera_view_matrix(), dtype=float)

    def _ui_profile_actions(self) -> None:  # pragma: no cover - Polyscope callback
        if self._psim.Button("Reset View"):
            self._reset_camera_view()
        self._psim.SameLine()
        if self._psim.Button("Save"):
            if self._session.save_as_new_enabled:
                self._save_as_new_profile()
            else:
                self._save_active_profile()

    def _ui_profile_save_as_new(self) -> None:  # pragma: no cover - Polyscope callback
        self._psim.SameLine()
        changed, save_as_new_enabled = ui_labeled_checkbox(
            self._psim,
            "As New",
            "##save_as_new",
            self._session.save_as_new_enabled,
        )
        if changed:
            self._session.save_as_new_enabled = save_as_new_enabled
            logger.debug("save_as_new_toggled", enabled=save_as_new_enabled)
            if self._session.save_as_new_enabled and not self._session.new_profile_name.strip():
                self._session.new_profile_name = self._default_profile_name()

        if self._session.save_as_new_enabled:
            name_width = ui_available_width(self._psim, min_width=140.0, max_width=260.0)
            with ui_labeled_item_width(self._psim, "Name", name_width):
                changed, new_profile_name = self._psim.InputText("##profile_name", self._session.new_profile_name)
            if changed:
                self._session.new_profile_name = new_profile_name
                logger.debug("new_profile_name_changed", profile_name=new_profile_name)

    def _ui_profile_status(self) -> None:  # pragma: no cover - Polyscope callback
        if self._session.profile_status_msg:
            self._psim.Text(self._session.profile_status_msg)
        if self._session.setting_unsaved:
            self._psim.Text("Unsaved changes")

    @freq_time_profiler("ui_info")
    def _ui_info_section(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self._psim, "Info", open_first_time=True) as expanded:
            if expanded:
                self._psim.Text(f"Version: {__version__}")
                self._psim.Text(f"FPS:   {self.fps:.1f}")
                self._psim.Text(f"Data:  {self.config.data_path.name}")
                self._psim.Text(f"Mesh:  {self.config.mesh_path.name}")

    @freq_time_profiler("ui_profile")
    def _ui_profile_section(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self._psim, "Profile", open_first_time=True) as expanded:
            if expanded:
                profile_width = ui_available_width(self._psim, min_width=160.0, max_width=280.0)
                with ui_labeled_item_width(self._psim, "Profile", profile_width):
                    profile_changed, next_profile = ui_select(
                        self._psim, "##profile", self.active_profile, list_profiles(self.root_dir)
                    )
                if profile_changed:
                    self._switch_profile(next_profile)

                self._ui_profile_actions()
                self._ui_profile_save_as_new()
                self._ui_profile_status()

    @freq_time_profiler("ui_appearance")
    def _ui_appearance_section(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self._psim, "Appearance", open_first_time=True) as expanded:
            if expanded:
                colormap_width, material_width = ui_equal_widths(self._psim, 2, min_width=90.0, max_width=150.0)
                with ui_labeled_item_width(self._psim, "Colormap", colormap_width):
                    colormap_changed, colormap = ui_select(
                        self._psim, "##colormap", self.settings.colormap, self._available_colormaps
                    )
                if colormap_changed:
                    self._set_settings(colormap=colormap)
                    self._refresh_quantities()

                self._psim.SameLine()
                with ui_labeled_item_width(self._psim, "Material", material_width):
                    material_changed, material = ui_select(
                        self._psim, "##material", self.settings.material, self._available_materials
                    )
                if material_changed:
                    self._set_settings(material=material)
                    self._refresh_quantities()

                effective_min, effective_max = self._effective_vminmax()
                color_min_width, color_max_width = ui_equal_widths(
                    self._psim,
                    2,
                    reserve=260.0,
                    min_width=70.0,
                    max_width=140.0,
                )
                with ui_labeled_item_width(self._psim, "Color Min", color_min_width):
                    min_changed, color_min = self._psim.InputFloat("##color_min", effective_min)
                self._psim.SameLine()
                with ui_labeled_item_width(self._psim, "Color Max", color_max_width):
                    max_changed, color_max = self._psim.InputFloat("##color_max", effective_max)
                self._psim.SameLine()
                if self._psim.Button("Auto Adjust"):
                    color_min, color_max = self._current_range()
                    min_changed = max_changed = True
                if min_changed or max_changed:
                    self._set_settings(color_min=color_min, color_max=color_max)
                    self._refresh_quantities()

    @freq_time_profiler("ui_processing")
    def _ui_processing_section(self) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self._psim, "Processing", open_first_time=True) as expanded:
            if expanded:
                transform_width = ui_available_width(self._psim, min_width=140.0, max_width=260.0)
                with ui_labeled_item_width(self._psim, "Transform", transform_width):
                    transform_changed, scalar_transform = ui_select(
                        self._psim,
                        "##scalar_transform",
                        self.settings.scalar_transform,
                        ["identity", "log_e", "log_10"],
                    )
                if transform_changed:
                    self._set_settings(scalar_transform=scalar_transform, color_min=None, color_max=None)
                    self._rebuild_texture()

                denoise_changed, denoise_enabled = ui_labeled_checkbox(
                    self._psim,
                    "Gaussian Denoise",
                    "##gaussian_denoise",
                    self.settings.denoise_enabled,
                )
                if denoise_changed:
                    self._set_settings(denoise_enabled=denoise_enabled)
                    self._rebuild_texture()

                self._psim.SameLine()
                sigma_width = ui_available_width(self._psim, min_width=90.0, max_width=140.0)
                with ui_labeled_item_width(self._psim, "Gaussian Sigma", sigma_width):
                    sigma_changed, denoise_sigma = self._psim.InputFloat(
                        "##gaussian_sigma", self.settings.denoise_sigma
                    )
                if sigma_changed:
                    self._set_settings(denoise_sigma=max(0.0, denoise_sigma))
                    self._rebuild_texture()
