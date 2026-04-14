# Usage Guide

## Launch

```bash
yumo2 gui --data path/to/field.plt --mesh path/to/mesh.stl
```

On Ubuntu under WSL, if the GUI segfaults immediately after opening, retry with
software OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 yumo2 gui --data path/to/field.plt --mesh path/to/mesh.stl
```

If GUI forwarding itself is not working, install the WSL packages listed in the
install guide first.

## Mesh Surface

The scalar field is baked onto the mesh via UV-unwrapping (xatlas) and trilinear
interpolation.  UV unwrapping is slow and runs once per mesh on startup.

### Colormap & Value Range

Select a colormap from the built-in list and adjust the color min/max, or reset
them to the full data range.

### Scalar Transform

Choose a transform in the **Processing** panel to apply to surface values
before colormapping: `identity` (default), `log_e`, or `log_10`.  Non-positive
values are replaced with the post-transform minimum.

### Denoising

Enable Gaussian denoising and adjust the sigma (in texels) to smooth the baked texture.

## Scope

Finds the min or max scalar value within a spherical radius.  Enable in the
**Scope** panel, set radius and mode, then click in the viewport to query.

## Coord Picker

Click on the mesh surface to read the scalar value at that point.  Expand the
**Coord Picker** panel to activate.

## Snapshot

Exports the current viewport as a PNG with an embedded colorbar.  Adjust the
crop region and colorbar position/size in the **Snapshot** panel, then click
**Save PNG**.

!!! tip "Live Preview and performance"

    Unchecking **Live Preview** freezes the preview and reduces CPU usage,
    giving higher and smoother frame rates during interactive work.

## Profiles

All settings are stored as named profiles in `.yumo2/profiles/`.  Use the
**Profile** panel to create, rename, or switch between them.
