# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

from __future__ import annotations

import numpy as np


def cross_nodes_edges(center: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    half = float(size) / 2.0
    cx, cy, cz = np.asarray(center, dtype=float)
    nodes = np.array(
        [
            [cx - half, cy, cz],
            [cx + half, cy, cz],
            [cx, cy - half, cz],
            [cx, cy + half, cz],
            [cx, cy, cz - half],
            [cx, cy, cz + half],
        ],
        dtype=float,
    )
    edges = np.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
        dtype=np.int32,
    )
    return nodes, edges
