# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog

from yumo2.colormap import generate_colorbar_rgba
from yumo2.imgui_image import NumpyImageTexture
from yumo2.profiling import freq_time_profiler
from yumo2.ui import ui_available_width, ui_image_window, ui_labeled_checkbox, ui_labeled_item_width, ui_tree_node

if TYPE_CHECKING:
    from yumo2.app import PolyscopeApp

_PREVIEW_MAX_HZ = 10.0

logger = structlog.get_logger(__name__)


def _flatten_rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    rgba_f = rgba.astype(np.float32) / 255.0
    alpha = rgba_f[:, :, 3:4]
    return rgba_f[:, :, :3] * alpha + (1.0 - alpha)


def _overlay_rgba(base_rgba: np.ndarray, overlay_rgba: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    composed = base_rgba.copy()
    base_h, base_w = composed.shape[:2]
    overlay_h, overlay_w = overlay_rgba.shape[:2]

    x1 = max(0, offset_x)
    y1 = max(0, offset_y)
    x2 = min(base_w, offset_x + overlay_w)
    y2 = min(base_h, offset_y + overlay_h)
    if x1 >= x2 or y1 >= y2:
        return composed

    overlay_x1 = x1 - offset_x
    overlay_y1 = y1 - offset_y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    base = composed[y1:y2, x1:x2]
    overlay = overlay_rgba[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    overlay_alpha = overlay[:, :, 3:4]
    base_alpha = base[:, :, 3:4]
    out_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)

    out_rgb = np.zeros_like(base[:, :, :3])
    visible = out_alpha > 1e-12
    out_rgb[visible[:, :, 0]] = (
        overlay[:, :, :3] * overlay_alpha + base[:, :, :3] * base_alpha * (1.0 - overlay_alpha)
    )[visible[:, :, 0]] / out_alpha[visible[:, :, 0]]

    composed[y1:y2, x1:x2, :3] = out_rgb
    composed[y1:y2, x1:x2, 3:4] = out_alpha
    return composed


class Snapshot:
    """Snapshot capture, preview, and export.

    Renders the current Polyscope viewport to a cropped, colorbar-composited
    image.  Maintains two ImGui image windows (overview with crop bounding-box,
    and a live preview of the final export) that are refreshed at up to
    ``_PREVIEW_MAX_HZ`` Hz.

    All crop/colorbar geometry is stored in :class:`~yumo2.settings.Settings`
    so parameters survive profile switches.
    """

    def __init__(self, app: PolyscopeApp) -> None:
        self.app = app
        self._overview_texture = NumpyImageTexture()
        self._preview_texture = NumpyImageTexture()

    def default_filename(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{ts}_{self.app.config.mesh_path.stem}_{self.app.config.data_path.stem}.png"

    def cleanup(self) -> None:
        self._overview_texture.release()
        self._preview_texture.release()

    def capture_rgba(self) -> np.ndarray:
        return self.app._ps.screenshot_to_buffer(transparent_bg=True)

    def crop_box(self, screen_h: int, screen_w: int) -> tuple[int, int, int, int]:
        cx = screen_w // 2 + self.app.settings.snapshot_crop_x
        cy = screen_h // 2 + self.app.settings.snapshot_crop_y
        x1 = max(0, min(cx - self.app.settings.snapshot_crop_w // 2, screen_w))
        y1 = max(0, min(cy - self.app.settings.snapshot_crop_h // 2, screen_h))
        x2 = max(0, min(x1 + self.app.settings.snapshot_crop_w, screen_w))
        y2 = max(0, min(y1 + self.app.settings.snapshot_crop_h, screen_h))
        return y1, x1, y2, x2

    def make_colorbar(self) -> np.ndarray:
        color_min, color_max = self.app._effective_vminmax()
        orientation: Literal["vertical", "horizontal"] = (
            "vertical"
            if self.app.settings.snapshot_colorbar_h >= self.app.settings.snapshot_colorbar_w
            else "horizontal"
        )
        return generate_colorbar_rgba(
            self.app.settings.snapshot_colorbar_h,
            self.app.settings.snapshot_colorbar_w,
            self.app.settings.colormap,
            color_min,
            color_max,
            self.app.settings.scalar_transform,
            self.app._loaded_cmaps,
            font_size=self.app.settings.snapshot_colorbar_font_size,
            orientation=orientation,
        )

    def overview(self, rgba: np.ndarray) -> np.ndarray:
        sh, sw = rgba.shape[:2]
        y1, x1, y2, x2 = self.crop_box(sh, sw)
        rgb = _flatten_rgba_on_white(rgba)
        border = 8
        rgb[y1 : y1 + border, x1:x2] = 0
        rgb[max(0, y2 - border) : y2, x1:x2] = 0
        rgb[y1:y2, x1 : x1 + border] = 0
        rgb[y1:y2, max(0, x2 - border) : x2] = 0
        return rgb

    def composed_rgba(self, rgba: np.ndarray) -> np.ndarray:
        sh, sw = rgba.shape[:2]
        y1, x1, y2, x2 = self.crop_box(sh, sw)
        crop = rgba[y1:y2, x1:x2].astype(np.float32) / 255.0
        return _overlay_rgba(
            crop,
            self.make_colorbar(),
            self.app.settings.snapshot_colorbar_x,
            self.app.settings.snapshot_colorbar_y,
        )

    def preview_rgb(self, rgba: np.ndarray) -> np.ndarray:
        composed = self.composed_rgba(rgba)
        alpha = composed[:, :, 3:4]
        return composed[:, :, :3] * alpha + (1.0 - alpha)

    def save(self) -> None:  # pragma: no cover - Polyscope runtime
        import cv2

        rgba = self.capture_rgba()
        composed = self.composed_rgba(rgba)
        bgra = cv2.cvtColor((composed * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
        filename = self.app._session.snapshot_filename.strip() or self.default_filename()
        path = Path(filename)
        cv2.imwrite(str(path), bgra)
        logger.info("snapshot_saved", path=str(path))
        self.app._session.snapshot_status_msg = f"Saved: {path.name}"
        self.app._session.snapshot_filename = self.default_filename()

    @freq_time_profiler("snapshot_capture")
    def _capture_preview_rgba(self) -> np.ndarray:  # pragma: no cover - Polyscope callback
        return self.capture_rgba()

    @freq_time_profiler("snapshot_compose")
    def _compose_preview_images(
        self, rgba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - Polyscope callback
        return self.overview(rgba), self.preview_rgb(rgba)

    @freq_time_profiler("snapshot_upload")
    def _upload_preview_images(
        self, overview: np.ndarray, preview: np.ndarray
    ) -> None:  # pragma: no cover - Polyscope callback
        self._overview_texture.update(overview)
        self._preview_texture.update(preview)

    def update_textures(self, rgba: np.ndarray) -> None:  # pragma: no cover - Polyscope callback
        overview, preview = self._compose_preview_images(rgba)
        self._upload_preview_images(overview, preview)

    def refresh_preview(self, now: float | None = None) -> None:  # pragma: no cover - Polyscope callback
        self.update_textures(self._capture_preview_rgba())
        self.app._session.snapshot_textures_initialized = True
        self.app._session.snapshot_last_preview_time = time.monotonic() if now is None else now

    def _should_refresh_preview(self, now: float) -> bool:
        if not self.app._session.snapshot_textures_initialized:
            return True
        if not self.app.settings.snapshot_live_preview:
            return False
        last = self.app._session.snapshot_last_preview_time
        if last is None:
            return True
        return (now - last) >= (1.0 / _PREVIEW_MAX_HZ)

    def draw_windows(self) -> None:  # pragma: no cover - Polyscope callback
        ui_image_window(
            self.app._psim,
            "snapshot_overview",
            self._overview_texture.texture_ref(),
            self._overview_texture.size(),
        )
        ui_image_window(
            self.app._psim,
            "snapshot_preview",
            self._preview_texture.texture_ref(),
            self._preview_texture.size(),
        )

    def _ui_image_controls(self, width: float) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self.app._psim, "Image", open_first_time=True) as image_expanded:
            if not image_expanded:
                return
            with ui_labeled_item_width(self.app._psim, "Crop H", width):
                changed, value = self.app._psim.InputInt("##croph", self.app.settings.snapshot_crop_h, 0)
            if changed:
                self.app._set_settings(snapshot_crop_h=max(1, value))
            self.app._psim.SameLine()
            with ui_labeled_item_width(self.app._psim, "Crop W", width):
                changed, value = self.app._psim.InputInt("##cropw", self.app.settings.snapshot_crop_w, 0)
            if changed:
                self.app._set_settings(snapshot_crop_w=max(1, value))

            with ui_labeled_item_width(self.app._psim, "Crop X", width):
                changed, value = self.app._psim.InputInt("##offx", self.app.settings.snapshot_crop_x, 0)
            if changed:
                self.app._set_settings(snapshot_crop_x=value)
            self.app._psim.SameLine()
            with ui_labeled_item_width(self.app._psim, "Crop Y", width):
                changed, value = self.app._psim.InputInt("##offy", self.app.settings.snapshot_crop_y, 0)
            if changed:
                self.app._set_settings(snapshot_crop_y=value)

    def _ui_colorbar_controls(self, width: float) -> None:  # pragma: no cover - Polyscope callback
        with ui_tree_node(self.app._psim, "Colorbar", open_first_time=True) as colorbar_expanded:
            if not colorbar_expanded:
                return
            with ui_labeled_item_width(self.app._psim, "Bar H", width):
                changed, value = self.app._psim.InputInt("##cbh", self.app.settings.snapshot_colorbar_h, 0)
            if changed:
                self.app._set_settings(snapshot_colorbar_h=max(1, value))
            self.app._psim.SameLine()
            with ui_labeled_item_width(self.app._psim, "Bar W", width):
                changed, value = self.app._psim.InputInt("##cbw", self.app.settings.snapshot_colorbar_w, 0)
            if changed:
                self.app._set_settings(snapshot_colorbar_w=max(1, value))

            with ui_labeled_item_width(self.app._psim, "Bar X", width):
                changed, value = self.app._psim.InputInt("##cbx", self.app.settings.snapshot_colorbar_x, 0)
            if changed:
                self.app._set_settings(snapshot_colorbar_x=value)
            self.app._psim.SameLine()
            with ui_labeled_item_width(self.app._psim, "Bar Y", width):
                changed, value = self.app._psim.InputInt("##cby", self.app.settings.snapshot_colorbar_y, 0)
            if changed:
                self.app._set_settings(snapshot_colorbar_y=value)

            with ui_labeled_item_width(self.app._psim, "Font", width):
                changed, value = self.app._psim.InputInt("##cbfont", self.app.settings.snapshot_colorbar_font_size, 0)
            if changed:
                self.app._set_settings(snapshot_colorbar_font_size=max(1, value))

    @freq_time_profiler("ui_snapshot")
    def ui(self) -> None:  # pragma: no cover - Polyscope callback
        now = time.monotonic()
        if self._should_refresh_preview(now):
            self.refresh_preview(now)
        self.draw_windows()

        with ui_tree_node(self.app._psim, "Snapshot", open_first_time=False) as expanded:
            if not expanded:
                return

            width = min(max(ui_available_width(self.app._psim) / 2.0, 90.0), 140.0)
            self._ui_image_controls(width)
            self._ui_colorbar_controls(width)
            filename_width = ui_available_width(self.app._psim, reserve=90.0, min_width=140.0, max_width=320.0)
            with ui_labeled_item_width(self.app._psim, "Filename", filename_width):
                changed, filename = self.app._psim.InputText("##snap", self.app._session.snapshot_filename)
            if changed:
                self.app._session.snapshot_filename = filename
                logger.debug("snapshot_filename_changed", filename=filename)

            changed, live = ui_labeled_checkbox(
                self.app._psim,
                "Live Preview",
                "##snapshot_live_preview",
                self.app.settings.snapshot_live_preview,
            )
            if changed:
                self.app._set_settings(snapshot_live_preview=live)

            self.app._psim.SameLine()
            if self.app._psim.Button("Refresh"):
                self.refresh_preview()

            self.app._psim.SameLine()

            if self.app._psim.Button("Save PNG"):
                self.save()
            if self.app._session.snapshot_status_msg:
                self.app._psim.Text(self.app._session.snapshot_status_msg)
