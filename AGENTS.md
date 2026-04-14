# AGENTS.md

This file provides guidance to coding agents working with this repository.

Claude-specific entry point: [CLAUDE.md](CLAUDE.md)

## Commands

Use `uv run` for all Python execution. Do not use bare `python` or `pip`.

```bash
# Install dependencies
make install

# Run all tests
make test

# Run a single test
uv run pytest tests/test_app.py::test_name -x

# Format and lint
make format

# Type check + pre-commit hooks
make check
```

## Architecture

`yumo2` is a Polyscope-based interactive tool for visualizing 3D scalar fields mapped onto mesh surfaces. It loads a scalar field (Tecplot `.plt` or plain text) and a mesh (`.stl`/`.obj`), UV-unwraps the mesh with xatlas, bakes the scalar field into a texture via trilinear interpolation, and renders it with Polyscope.

### Core Data Flow

```text
CLI (typer/__main__.py)
  -> PolyscopeApp.run()
      -> _load_mesh_and_uv()      # trimesh + xatlas (slow, one-time)
      -> _load_data()             # parse scalar field into 3D numpy grid
      -> _init_polyscope()        # load assets (HDR materials, colormaps)
      -> _setup_scene()           # bake texture, register mesh, set camera
      -> ps.show()                # Polyscope event loop
           -> callback() each frame  # renders all ImGui UI panels
```

### Key Classes

`PolyscopeApp` (`app.py`) is the central orchestrator. It holds three main state objects:
- `Config`: immutable input file paths
- `Settings`: a Pydantic model persisted as JSON profiles under `.yumo2/profiles/`
- `SessionContext`: transient per-session state such as mesh data, baked textures, and UI state

Extension hooks for subclassing:
- `pre_load_hook`
- `post_load_hook`
- `post_polyscope_init_hook`
- `post_scene_setup_hook`

`SessionContext` (`app.py`) contains transient state so profile switching via `_reset_session_context()` can cleanly reset the session. If a feature needs per-session state, put it in `SessionContext` instead of storing it on the feature object.

Features in `yumo2/features/`:
- `Snapshot`
- `Scope`
- `Picker`

Each feature is instantiated once in `PolyscopeApp.__init__` and exposes a `ui()` method called every frame from `callback()`.

### Settings Persistence

`Settings` is a Pydantic model. `_set_settings(**kwargs)` applies changes and immediately writes the active profile to disk. Profiles are stored in `.yumo2/profiles/<name>.json`; the active profile name is tracked in `.yumo2/state.json`.

### Profiling

Set `YUMO2_PROFILE=time` (or `memory`, `all`) to enable profiling. `freq_time_profiler` accumulates samples over a time window and logs windowed stats. `time_profiler` and `mem_profiler` are used for one-shot measurements.

### Testing

Tests avoid real Polyscope/ImGui dependencies by injecting fake `_ps` and `_psim` objects directly onto `PolyscopeApp`. UI callback methods marked with `# pragma: no cover - Polyscope callback` are excluded from coverage. Use `tmp_path` fixtures for file I/O tests.
