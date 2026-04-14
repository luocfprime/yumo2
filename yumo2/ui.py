# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from contextlib import contextmanager


@contextmanager
def ui_item_width(psim, width):
    psim.PushItemWidth(width)
    try:
        yield
    finally:
        psim.PopItemWidth()


@contextmanager
def ui_tree_node(psim, label, open_first_time=True):
    """
    A context manager for creating a collapsible ImGui tree node.

    This correctly handles the ImGui pattern of checking the return of TreeNode
    and calling TreePop conditionally, while ensuring the context manager
    protocol is always followed.
    """
    psim.SetNextItemOpen(open_first_time, psim.ImGuiCond_FirstUseEver)
    expanded = psim.TreeNode(label)

    try:
        yield expanded
    finally:
        if expanded:
            psim.TreePop()


def ui_select(psim, label, current, options):
    changed = False
    value = current
    expanded = psim.BeginCombo(label, current)
    try:
        if not expanded:
            return changed, value
        for option in options:
            selected = psim.Selectable(option, option == current)
            if selected and option != current:
                value = option
                changed = True
        return changed, value
    finally:
        if expanded:
            psim.EndCombo()


def ui_radio_select(psim, label, current, options, *, horizontal=True):
    changed = False
    value = current

    ui_label(psim, label)
    for index, option in enumerate(options):
        clicked = psim.RadioButton(option, option == value)
        if clicked and option != current:
            value = option
            changed = True
        if horizontal and index < len(options) - 1:
            psim.SameLine()

    return changed, value


def ui_label(psim, label):
    psim.AlignTextToFramePadding()
    psim.Text(label)
    psim.SameLine()


@contextmanager
def ui_labeled_item_width(psim, label, width):
    ui_label(psim, label)
    with ui_item_width(psim, width):
        yield


def ui_labeled_checkbox(psim, label, hidden_label, value):
    ui_label(psim, label)
    return psim.Checkbox(hidden_label, value)


def ui_available_width(psim, *, reserve: float = 0.05, min_width: float = 1.0, max_width: float | None = None) -> float:
    avail_w, _ = psim.GetContentRegionAvail()
    usable_w = max(float(avail_w) * 0.95 - reserve, 0.0)
    width = max(usable_w, min_width)
    if max_width is not None:
        width = min(width, max_width)
    return width


def ui_equal_widths(
    psim,
    count: int,
    *,
    spacing: float = 8.0,
    reserve: float = 0.05,
    min_width: float = 1.0,
    max_width: float | None = None,
) -> tuple[float, ...]:
    avail_w, _ = psim.GetContentRegionAvail()
    usable_w = max(float(avail_w) * 0.95 - reserve, 0.0)
    width = max((usable_w - spacing * max(0, count - 1)) / max(count, 1), min_width)
    if max_width is not None:
        width = min(width, max_width)
    return tuple(width for _ in range(count))


def ui_image_window(psim, name, texture_ref, size, max_initial_size: int = 400):
    width, height = size
    width_f = float(width)
    height_f = float(height)
    scale = min(1.0, max_initial_size / max(width_f, height_f, 1.0))
    psim.SetNextWindowSize((width_f * scale, height_f * scale), psim.ImGuiCond_FirstUseEver)
    if psim.Begin(name):
        avail_w, _ = psim.GetContentRegionAvail()
        draw_w = min(width_f, float(avail_w)) if avail_w > 0 else width_f
        aspect = height_f / width_f if width_f > 0 else 1.0
        draw_h = draw_w * aspect
        psim.Image(texture_ref, (draw_w, draw_h))
    psim.End()
