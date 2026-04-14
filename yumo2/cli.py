# SPDX-FileCopyrightText: 2025 Chaofan Luo
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Yumo2-Commercial
# Commercial licensing available; see LICENSES/LicenseRef-Yumo2-Commercial.txt.

"""yumo2 CLI — app instance, built-in commands, and plugin loading."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated

import typer

from yumo2.__about__ import __version__
from yumo2.app import Config, PolyscopeApp
from yumo2.log import configure_structlog

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


@app.callback()
def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the application version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """yumo2 CLI."""
    configure_structlog()


@app.command()
def gui(
    data_path: Annotated[
        Path,
        typer.Option(
            "--data",
            "-d",
            help="Path to data file (e.g. Tecplot .plt file)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    mesh_path: Annotated[
        Path,
        typer.Option(
            "--mesh",
            "-m",
            help="Path to mesh file (e.g. .stl file)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Open the interactive GUI for visualising a scalar field mapped onto a mesh."""
    PolyscopeApp(Config(data_path=data_path, mesh_path=mesh_path)).run()


def _load_plugins() -> None:
    for ep in entry_points(group="yumo2.plugins"):
        module_name = ep.value.split(":")[0]
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            typer.echo(f"[warn] plugin '{ep.name}' failed to load: {exc}", err=True)


def main() -> None:
    _load_plugins()
    app()
