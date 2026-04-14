# Install

**Prerequisite:** [uv](https://docs.astral.sh/uv/getting-started/installation/) must be installed.

Install from PyPI:

```bash
uv tool install yumo2
```

Or run once without installing:

```bash
uv run --with yumo2 yumo2 gui --data field.plt --mesh mesh.stl
```

## Ubuntu on WSL

Install the common X11 / OpenGL runtime packages first:

```bash
sudo apt update
sudo apt install -y libglu1-mesa mesa-utils x11-apps
```

Verify that GUI forwarding is available:

```bash
echo $DISPLAY
xeyes
```

If the GUI opens a black window briefly and then segfaults under WSL, run yumo2
with software OpenGL:

```bash
uv tool install yumo2
LIBGL_ALWAYS_SOFTWARE=1 yumo2 gui --data field.plt --mesh mesh.stl
```

Or run once without installing:

```bash
LIBGL_ALWAYS_SOFTWARE=1 uv run --with yumo2 yumo2 gui --data field.plt --mesh mesh.stl
```

`mesa-utils` is optional for diagnostics such as `glxinfo`, and `x11-apps` is
useful for quickly verifying that X11/WSLg forwarding is working.
