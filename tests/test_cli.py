# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

import logging

from typer.testing import CliRunner

from yumo2.__about__ import __version__
from yumo2.__main__ import app as cli_app
from yumo2.log import configure_structlog

runner = CliRunner()


def test_cli_version_flag_prints_version() -> None:
    result = runner.invoke(cli_app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_configure_structlog_reads_log_level_env(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    configure_structlog()

    assert logging.getLogger().level == logging.DEBUG
