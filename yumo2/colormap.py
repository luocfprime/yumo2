# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

logger = structlog.get_logger(__name__)

_CMAP_CACHE: dict[str, Colormap] = {}
_font_warning_logged = False
_LOADED_PS_COLORMAPS: set[str] = set()


def load_colormaps(ps, *directories: Path | None) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for directory in directories:
        if directory is None or not directory.exists():
            continue

        for path in sorted(directory.glob("*_colormap.png")):
            name = path.stem.removesuffix("_colormap")
            if name in _LOADED_PS_COLORMAPS:
                logger.debug("colormap_already_loaded", name=name, path=str(path))
                loaded[name] = str(path)
                continue
            try:
                ps.load_color_map(name, str(path))
                loaded[name] = str(path)
                _LOADED_PS_COLORMAPS.add(name)
            except Exception as exc:  # pragma: no cover - depends on Polyscope runtime
                logger.warning("failed_to_load_colormap", path=str(path), error=str(exc))

    return loaded


def _get_cmap(name: str, loaded_cmaps: dict[str, str] | None = None) -> Colormap:
    if name in _CMAP_CACHE:
        return _CMAP_CACHE[name]

    if loaded_cmaps and name in loaded_cmaps:
        image = Image.open(loaded_cmaps[name]).convert("RGB")
        data = np.asarray(image, dtype=np.float32) / 255.0
        center_row = data[data.shape[0] // 2]
        colors = [tuple(center_row[index]) for index in range(center_row.shape[0])]
        cmap = LinearSegmentedColormap.from_list(name, colors, N=len(colors))
    else:
        cmap = plt.get_cmap(name)  # type: ignore[assignment]

    _CMAP_CACHE[name] = cmap
    return cmap


_FONT_CANDIDATES = [
    "Arial.ttf",
    "arial.ttf",
    "/Library/Fonts/Arial.ttf",  # macOS system
    "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
    "C:/Windows/Fonts/arial.ttf",  # Windows
]


def _load_font(font: str, font_size: int) -> ImageFont.FreeTypeFont:
    global _font_warning_logged

    for candidate in [font, *_FONT_CANDIDATES]:
        try:
            return ImageFont.truetype(candidate, size=font_size)
        except OSError:
            continue

    if not _font_warning_logged:
        _font_warning_logged = True
        logger.warning("could_not_load_any_font_using_bitmap_fallback")
    return ImageFont.load_default(size=font_size)  # type: ignore[return-value]


def _format_tick(value: float, method: str) -> str:
    if method == "identity":
        return f"{value:.2g}"
    if method == "log_e":
        return f"e^{value:.2g}"
    if method == "log_10":
        exponent = int(np.floor(value))
        coefficient = 10 ** (value - exponent)
        return f"{coefficient:.2f}e{exponent}"
    raise ValueError(f"Unsupported scalar transform: {method}")


def _make_labels(values: np.ndarray, method: str) -> list[str]:
    labels: list[str] = []
    for index, value in enumerate(values):
        formatted = _format_tick(value, method)
        if index == 0:
            labels.append(f">= {formatted}")
        elif index == len(values) - 1:
            labels.append(f"<= {formatted}")
        else:
            labels.append(formatted)
    return labels


def generate_colorbar_image(
    colorbar_height: int,
    colorbar_width: int,
    cmap: str,
    c_min: float,
    c_max: float,
    method: str = "identity",
    loaded_cmaps: dict[str, str] | None = None,
    font: str = "Arial.ttf",
    font_size: int = 12,
    orientation: Literal["vertical", "horizontal"] = "vertical",
) -> np.ndarray:
    return generate_colorbar_rgba(
        colorbar_height=colorbar_height,
        colorbar_width=colorbar_width,
        cmap=cmap,
        c_min=c_min,
        c_max=c_max,
        method=method,
        loaded_cmaps=loaded_cmaps,
        font=font,
        font_size=font_size,
        orientation=orientation,
    )[:, :, :3]


def generate_colorbar_rgba(
    colorbar_height: int,
    colorbar_width: int,
    cmap: str,
    c_min: float,
    c_max: float,
    method: str = "identity",
    loaded_cmaps: dict[str, str] | None = None,
    font: str = "Arial.ttf",
    font_size: int = 12,
    orientation: Literal["vertical", "horizontal"] = "vertical",
) -> np.ndarray:
    font_obj = _load_font(font, font_size)
    tick_values = np.linspace(c_max, c_min, 7)
    labels = _make_labels(tick_values, method)

    bar_width = 25
    text_padding = 15
    tick_spacing = 10
    colormap = _get_cmap(cmap, loaded_cmaps)

    if orientation == "vertical":
        max_label_width = max(font_obj.getbbox(label)[2] for label in labels)
        image_width = max(colorbar_width, bar_width + tick_spacing + max_label_width + 2 * text_padding)

        image = Image.new("RGBA", (int(image_width), int(colorbar_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        bar_start_y = text_padding
        bar_end_y = colorbar_height - text_padding
        bar_height = max(1, bar_end_y - bar_start_y)
        gradient = np.linspace(1.0, 0.0, bar_height)
        colors = (colormap(gradient)[:, :3] * 255).astype(np.uint8)

        for row_index, color in enumerate(colors):
            y = bar_start_y + row_index
            draw.line([(text_padding, y), (text_padding + bar_width, y)], fill=(*tuple(color), 255))

        tick_positions = np.linspace(bar_start_y, bar_end_y, len(labels))
        for label, position in zip(labels, tick_positions, strict=False):
            draw.line(
                [(text_padding + bar_width, position), (text_padding + bar_width + 5, position)],
                fill=(0, 0, 0, 255),
            )
            _, _, _, text_height = font_obj.getbbox(label)
            draw.text(
                (text_padding + bar_width + tick_spacing, int(position - text_height / 2)),
                label,
                fill=(0, 0, 0, 255),
                font=font_obj,
            )
    else:
        max_label_height = max(font_obj.getbbox(label)[3] for label in labels)
        image_height = max(colorbar_height, bar_width + tick_spacing + max_label_height + 2 * text_padding)

        image = Image.new("RGBA", (int(colorbar_width), int(image_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        bar_start_x = text_padding
        bar_end_x = colorbar_width - text_padding
        bar_length = max(1, bar_end_x - bar_start_x)
        gradient = np.linspace(0.0, 1.0, bar_length)
        colors = (colormap(gradient)[:, :3] * 255).astype(np.uint8)

        bar_y = text_padding
        for col_index, color in enumerate(colors):
            x = bar_start_x + col_index
            draw.line([(x, bar_y), (x, bar_y + bar_width)], fill=(*tuple(color), 255))

        tick_positions = np.linspace(bar_start_x, bar_end_x, len(labels))
        for label, position in zip(labels, tick_positions, strict=False):
            draw.line(
                [(position, bar_y + bar_width), (position, bar_y + bar_width + 5)],
                fill=(0, 0, 0, 255),
            )
            left, _, right, _ = font_obj.getbbox(label)
            text_width = right - left
            draw.text(
                (int(position - text_width / 2), bar_y + bar_width + tick_spacing),
                label,
                fill=(0, 0, 0, 255),
                font=font_obj,
            )

    return np.asarray(image, dtype=np.float32) / 255.0
