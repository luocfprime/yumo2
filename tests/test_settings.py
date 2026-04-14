# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from pathlib import Path

import pytest

from yumo2.settings import Settings, list_profiles, load_active_profile, load_profile, save_profile


def test_load_active_profile_creates_default_profile(tmp_path: Path) -> None:
    profile_name, settings = load_active_profile(tmp_path)

    assert profile_name == "default"
    assert settings == Settings()
    assert settings.colormap == "RdBu"
    assert settings.material == "0.30_clay_0.70_flat"
    assert (tmp_path / ".yumo2" / "profiles" / "default.json").exists()
    assert (tmp_path / ".yumo2" / "state.json").exists()


def test_save_and_list_profiles_round_trip(tmp_path: Path) -> None:
    save_profile("zeta", Settings(denoise_sigma=2.5), tmp_path)
    save_profile("alpha", Settings(colormap="magma"), tmp_path)

    assert list_profiles(tmp_path) == ["alpha", "zeta"]
    assert load_profile("alpha", tmp_path).colormap == "magma"
    assert load_profile("zeta", tmp_path).denoise_sigma == 2.5


def test_load_active_profile_uses_most_recent_profile(tmp_path: Path) -> None:
    save_profile("default", Settings(), tmp_path)
    save_profile("experiment", Settings(scalar_transform="log_10"), tmp_path)

    profile_name, settings = load_active_profile(tmp_path)

    assert profile_name == "experiment"
    assert settings.scalar_transform == "log_10"


def test_load_profile_accepts_legacy_preprocess_method_key(tmp_path: Path) -> None:
    import json

    from yumo2.settings import ensure_store, profile_path

    ensure_store(tmp_path)
    legacy_json = json.dumps({"preprocess_method": "log_e", "denoise_sigma": 1.5})
    profile_path("legacy", tmp_path).write_text(legacy_json)

    settings = load_profile("legacy", tmp_path)

    assert settings.scalar_transform == "log_e"
    assert settings.denoise_sigma == 1.5


@pytest.mark.parametrize(
    "settings",
    [
        Settings(
            camera_view=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            snapshot_crop_h=512,
            snapshot_crop_w=768,
            snapshot_crop_x=12,
            snapshot_crop_y=-5,
            snapshot_colorbar_h=240,
            snapshot_colorbar_w=96,
            snapshot_colorbar_x=20,
            snapshot_colorbar_y=24,
            snapshot_colorbar_font_size=18,
            scope_enabled=False,
            scope_mode="Min",
            scope_radius=2.5,
            scope_marker_length_px=42.0,
            scope_marker_thickness_px=5.0,
            scope_marker_max_fraction=0.03,
        )
    ],
)
def test_settings_round_trip_with_profile(tmp_path: Path, settings: Settings) -> None:
    save_profile("default", settings, tmp_path)

    loaded = load_profile("default", tmp_path)

    assert loaded == settings
