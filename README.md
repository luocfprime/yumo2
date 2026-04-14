# yumo2

Interactive 3D visualization tool for mapping scalar fields onto mesh surfaces.

This project is dual-licensed under `AGPL-3.0-or-later` and a separate
commercial license.

- **Repository**: <https://github.com/luocfprime/yumo2/>
- **Documentation**: <https://luocfprime.github.io/yumo2/>

## What it does

yumo2 loads a volumetric scalar field (Tecplot `.plt`) and a 3D mesh (`.stl`, `.obj`, etc.), UV-unwraps the mesh, bakes the scalar field onto it via trilinear interpolation, and renders the result interactively with [Polyscope](https://polyscope.run/).

Features:
- Colormap selection with adjustable color range
- Log/linear transformation of scalar values
- Gaussian denoising of the baked texture
- Interactive scope tool (find min/max value within a radius)
- Snapshot export with embedded colorbar

## Installation

```bash
uv tools install yumo2
```

Requires [uv](https://docs.astral.sh/uv/).

## Usage

```bash
yumo2 gui --data path/to/field.plt --mesh path/to/mesh.stl
```

## Development

```bash
make test      # run tests
make format    # ruff format + lint
make check     # pre-commit hooks + mypy
```

Run a single test:

```bash
uv run pytest tests/test_app.py::test_name -x
```

## License

`yumo2` is available under a dual-license model:

- `AGPL-3.0-or-later` for open source use
- A commercial license for proprietary products, internal deployments, and
  SaaS use without AGPL obligations

See [AGPL text](LICENSES/AGPL-3.0-or-later.txt) and
[commercial license](LICENSES/LicenseRef-Yumo2-Commercial.txt).
