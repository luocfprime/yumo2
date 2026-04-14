# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from yumo2.ui import ui_image_window


def test_ui_image_window_scales_to_available_width_and_preserves_aspect_ratio() -> None:
    class FakePsim:
        ImGuiCond_FirstUseEver = 4

        def __init__(self) -> None:
            self.calls = []

        def SetNextWindowSize(self, size, cond) -> None:
            self.calls.append(("SetNextWindowSize", size, cond))

        def Begin(self, name: str) -> bool:
            self.calls.append(("Begin", name))
            return True

        def GetContentRegionAvail(self):
            return (100.0, 500.0)

        def Image(self, texture_ref, image_size) -> None:
            self.calls.append(("Image", texture_ref, image_size))

        def End(self) -> None:
            self.calls.append(("End",))

    psim = FakePsim()

    ui_image_window(psim, "snapshot_preview", "tex", (200, 100))

    assert psim.calls == [
        ("SetNextWindowSize", (200.0, 100.0), 4),
        ("Begin", "snapshot_preview"),
        ("Image", "tex", (100.0, 50.0)),
        ("End",),
    ]


def test_ui_available_width_leaves_default_margin() -> None:
    class FakePsim:
        @staticmethod
        def GetContentRegionAvail():
            return (200.0, 100.0)

    from yumo2.ui import ui_available_width

    assert ui_available_width(FakePsim()) == 189.95


def test_ui_equal_widths_leave_default_margin() -> None:
    class FakePsim:
        @staticmethod
        def GetContentRegionAvail():
            return (200.0, 100.0)

    from yumo2.ui import ui_equal_widths

    assert ui_equal_widths(FakePsim(), 2, spacing=10.0) == (89.975, 89.975)
