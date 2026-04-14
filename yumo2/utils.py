# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

import numpy as np


def transform_scalar_data(values: np.ndarray, method: str) -> np.ndarray:
    if method == "identity":
        return values.copy()
    if method in ("log_e", "log_10"):
        transformed = values.copy()
        log_fn = np.log if method == "log_e" else np.log10
        nonzero_mask = transformed > 0

        if not np.any(nonzero_mask):
            raise ValueError(f"No positive values found for {method} transform.")

        transformed[nonzero_mask] = log_fn(transformed[nonzero_mask])
        min_value = np.min(transformed[nonzero_mask])
        transformed[~nonzero_mask] = min_value
        return transformed
    raise ValueError(f"Unknown scalar transform: {method}")


def finite_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Expected at least one finite value")
    return float(np.min(finite)), float(np.max(finite))
