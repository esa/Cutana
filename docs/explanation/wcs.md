[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# WCS handling

## FITS output

FITS cutouts keep the full WCS of their parent tile, with accurate astrometric calibration:

- **Parent WCS reproduction**: the cutout reproduces the parent tile's mapping exactly. `CRVAL`, `CTYPE` and the CD/PC orientation are inherited unchanged, and only `CRPIX` is shifted to the extraction origin. When the parent tile carries no usable WCS, Cutana synthesises a minimal TAN header instead, with `CRPIX` at the cutout centre and `CRVAL` at the source position.
- **Pixel scale correction**: when a cutout is resized from its extracted size to `target_resolution`, the WCS pixel scale is adjusted so coordinates still map correctly.
- **Sky area preservation**: the sky area a cutout covers stays the same; only the pixel scale changes with the resize.
- **Format compatibility**: CD matrix, CDELT and PC+CDELT WCS formats in the parent files are all supported.

Pixel units are separate from the WCS; see [pixel units in FITS headers](../reference/output_formats.md#pixel-units-in-fits-headers).

!!! note "Why the parent mapping is inherited"
    Re-projecting a tangent plane at each source's position rotates the cutout frame by the meridian convergence between tile centre and source. The error is zero at the cutout centre and grows towards the edges, and it is worst far from the tile centre and at high declination. Inheriting the parent's mapping avoids it. Each release checks cutout WCS against the parent tiles to keep it that way.

## Zarr output

!!! warning
    Zarr archives **do not contain WCS information**. The metadata records each source's position and size, and where in the parent tile the cutout was extracted; see the [Zarr metadata columns](../reference/output_formats.md#zarr).
