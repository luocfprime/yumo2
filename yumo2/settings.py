# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import structlog
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from yumo2.constants import DEFAULT_COLORMAP, DEFAULT_MATERIAL

logger = structlog.get_logger(__name__)


class Settings(BaseModel):
    """Immutable user-facing settings persisted as a named profile.

    All UI-controllable parameters live here.  Instances are created by
    :func:`load_active_profile` / :func:`load_profile` and written to disk by
    :func:`save_profile`.  ``PolyscopeApp._set_settings`` is the only code path
    that should mutate the active settings — it creates a new instance and saves
    it atomically.
    """

    model_config = ConfigDict(frozen=True)
    colormap: str = DEFAULT_COLORMAP
    color_min: float | None = None
    color_max: float | None = None
    scalar_transform: Literal["identity", "log_e", "log_10"] = Field(
        default="log_10",
        validation_alias=AliasChoices("scalar_transform", "preprocess_method"),
    )
    material: str = DEFAULT_MATERIAL
    denoise_enabled: bool = True
    denoise_sigma: float = 3.0
    camera_view: list[list[float]] | None = None
    snapshot_crop_h: int = 600
    snapshot_crop_w: int = 800
    snapshot_crop_x: int = 0
    snapshot_crop_y: int = 0
    snapshot_colorbar_h: int = 300
    snapshot_colorbar_w: int = 120
    snapshot_colorbar_x: int = 10
    snapshot_colorbar_y: int = 10
    snapshot_colorbar_font_size: int = 12
    snapshot_live_preview: bool = False
    scope_enabled: bool = True
    scope_mode: Literal["Min", "Max"] = "Max"
    scope_radius: float = 0.5
    scope_marker_length_px: float = 30.0
    scope_marker_thickness_px: float = 3.0
    scope_marker_max_fraction: float = 0.02


def yumo_dir(root: Path | str = ".") -> Path:
    return Path(root) / ".yumo2"


def profiles_dir(root: Path | str = ".") -> Path:
    return yumo_dir(root) / "profiles"


def state_path(root: Path | str = ".") -> Path:
    return yumo_dir(root) / "state.json"


def ensure_store(root: Path | str = ".") -> None:
    profiles_dir(root).mkdir(parents=True, exist_ok=True)


def _read_state(root: Path | str = ".") -> dict:
    path = state_path(root)
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    return state if isinstance(state, dict) else {}


def _write_state(state: dict, root: Path | str = ".") -> None:
    ensure_store(root)
    state_path(root).write_text(json.dumps(state, indent=2))


def list_profiles(root: Path | str = ".") -> list[str]:
    ensure_store(root)
    return sorted(path.stem for path in profiles_dir(root).glob("*.json"))


def profile_path(name: str, root: Path | str = ".") -> Path:
    return profiles_dir(root) / f"{name}.json"


def load_profile(name: str, root: Path | str = ".") -> Settings:
    path = profile_path(name, root)
    logger.info("loading_profile", profile=name, path=str(path))
    return Settings.model_validate_json(path.read_text())


def _write_active_profile(name: str, root: Path | str = ".") -> None:
    state = _read_state(root)
    state["active_profile"] = name
    _write_state(state, root)


def save_profile(name: str, settings: Settings, root: Path | str = ".") -> None:
    ensure_store(root)
    path = profile_path(name, root)
    path.write_text(settings.model_dump_json(indent=2))
    _write_active_profile(name, root)
    logger.info("profile_saved", profile=name, path=str(path))


def _read_active_profile_name(root: Path | str = ".") -> str | None:
    state = _read_state(root)
    active_profile = state.get("active_profile")
    return active_profile if isinstance(active_profile, str) else None


def set_active_profile(name: str, root: Path | str = ".") -> None:
    if not profile_path(name, root).exists():
        raise FileNotFoundError(f"Profile does not exist: {name}")
    _write_active_profile(name, root)


def load_active_profile(root: Path | str = ".") -> tuple[str, Settings]:
    ensure_store(root)

    active_profile = _read_active_profile_name(root)
    if active_profile and profile_path(active_profile, root).exists():
        return active_profile, load_profile(active_profile, root)

    existing_profiles = list_profiles(root)
    if existing_profiles:
        active_profile = existing_profiles[0]
        _write_active_profile(active_profile, root)
        return active_profile, load_profile(active_profile, root)

    active_profile = "default"
    settings = Settings()
    save_profile(active_profile, settings, root)
    return active_profile, settings
