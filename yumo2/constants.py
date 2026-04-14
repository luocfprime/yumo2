# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

ASSETS_ROOT = Path(__file__).parent / "assets"
DEFAULT_IMGUI_INI = ASSETS_ROOT / "default_imgui.ini"

SCALAR_TRANSFORM_METHODS = ("identity", "log_e", "log_10")

DEFAULT_COLORMAP = "RdBu"
DEFAULT_MATERIAL = "0.30_clay_0.70_flat"

BUILTIN_COLORMAPS = ("viridis", "magma", "turbo", "coolwarm", "jet", "rainbow")
BUILTIN_MATERIALS = ("clay", "wax", "candy", "flat", "mud", "ceramic", "jade", "normal")
