# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

"""Module entry point with backwards-compatible CLI exports."""

from yumo2.cli import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
