[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Keep physical units

By default, Cutana converts pixels to Jansky and then normalises them for display, which discards the physical scale. This page shows how to keep calibrated values.

## Keep pixels in Jansky

Flux conversion to Jy is on by default and uses the `MAGZERO` header keyword (configurable via `config.flux_conversion_keywords.AB_zeropoint`). Normalisation stretches pixels into the `data_type` range, so turn it off and write floats:

```python
config.normalisation_method = "none"  # or config.do_only_cutout_extraction = True
config.data_type = "float32"  # uint8 rescales to 0-255, losing the scale
```

To keep the parent tile's own units instead, disable the conversion:

```python
config.apply_flux_conversion = False
```

## Conserve flux when resizing

For photometry, resize with flux conservation:

```python
config.flux_conserved_resizing = True
config.data_type = "float32"  # preserve numerical precision
config.normalisation_method = "none"  # keep the original flux values
```

This preserves total flux through the resize. It uses [drizzle](https://github.com/spacetelescope/drizzle), which is slower than the default resizing.

## Extract raw cutouts

Set `do_only_cutout_extraction = True` with `output_format = "fits"` to skip resizing, normalisation and channel combination. Only the flux conversion is applied, and you can disable that too. The output `data_type` follows the input. This mode does not support Zarr output.

Check the result with the `BUNIT`, `FLUXAPPX`, `UNIT` and `CONSVFLX` header keywords, described in the [output formats reference](../reference/output_formats.md#pixel-units-in-fits-headers).
