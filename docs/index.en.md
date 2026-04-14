# yumo2

3D scalar field visualization on mesh surfaces.

<video width=100% autoplay muted loop>
  <source src="[[url.videos]]/media/movies/teaser.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

yumo2 loads a volumetric scalar field and maps it onto a triangular mesh via UV-unwrapping and trilinear interpolation. The result is rendered interactively with [Polyscope](https://polyscope.run/).

## Features

- Colormap selection with adjustable color range
- Log/linear scalar transform of surface values
- Gaussian denoising of the baked texture
- Interactive scope tool: find the min/max value within a radius
- Snapshot export with embedded colorbar
