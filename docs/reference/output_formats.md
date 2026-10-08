[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Output formats

Set `output_format` to `"zarr"` (default) or `"fits"`.

| Format | Best for | WCS | Raw cutouts (`do_only_cutout_extraction`) |
| --- | --- | --- | --- |
| Zarr | Large datasets and analysis or ML workflows | Not stored | Not supported |
| FITS | Compatibility with existing astronomical software | Full WCS per cutout | Required |

## Zarr

Cutouts are stored in [Zarr](https://zarr.readthedocs.io/en/stable/) archives, written with [images_to_zarr](https://github.com/gomezzz/images_to_zarr/).

Output is organised in batches. Each batch gets a folder named `batch_cutout_process_{index}_{unique_id}` (`batch_streaming_…` for the streaming backend), containing:

| File | Content |
| --- | --- |
| `images.zarr` | Array `images` of shape `(n_images, H, W, C)` |
| `images_metadata.parquet` | One row per image, mapping it to its source |

Row *i* of the metadata describes `images[i]` in the archive next to it. The columns are:

| Column | Meaning |
| --- | --- |
| `source_id` | The catalogue `SourceID` |
| `ra`, `dec` | Source position in degrees |
| `diameter_pixel`, `diameter_arcsec` | Source size as given in the catalogue; the other one is empty |
| `tile` | FITS file the cutout was extracted from |
| `pixel_scale_arcsec_per_pixel` | Pixel scale of the resized cutout |
| `original_cutout_size` | Extracted size in parent pixels, before resizing |
| `extraction_origin_x`, `extraction_origin_y`, `extraction_size` | Where in the parent tile the cutout was extracted |
| `rescaled_offset_x`, `rescaled_offset_y` | Sub-pixel offset of the source from the cutout centre after resizing |
| `processing_timestamp` | When the batch was processed (Unix time) |

!!! note
    The Zarr archives store no source IDs and no WCS. Use the metadata's row order to find a source's cutout, as shown in [read Zarr outputs](../how_to/read_outputs.md). The metadata records the centre and the size in pixels or arcseconds, depending on what the catalogue provided.

## FITS

Each source gets its own FITS file, with its information in the header of the `PRIMARY` extension and a full WCS; see [WCS handling](../explanation/wcs.md).

### Pixel units in FITS headers

FITS cutouts record their pixel unit in these keywords:

| Keyword | HDU | Meaning |
| --- | --- | --- |
| `BUNIT` | image | Standard, machine-readable unit. Written only when Cutana knows the unit, which means `Jy`; absent in every other case |
| `FLUXAPPX` | image | `T` when the resize did not conserve flux, so the values only approximate `BUNIT`. Written only alongside `BUNIT` |
| `UNIT` | primary | Deprecated 0.3.2 keyword, kept for existing readers. Prefer `BUNIT` |
| `CONSVFLX` | primary | Whether flux-conserved resizing was used |

`UNIT` takes one of these values, prefixed `approx ` when the resize did not conserve flux:

| `UNIT` | When | `BUNIT` |
| --- | --- | --- |
| `Jy` | Flux conversion on, pixels still on their physical scale | `Jy` |
| `OriginalUnit` | Flux conversion off, so pixels keep the parent tile's unit, which Cutana cannot name without reading it back from the parent header | absent |
| `UserConversionUnit` | `config.user_flux_conversion_function` replaced the AB-zeropoint maths, so the pixels were converted, but not by Cutana and not necessarily to Jy | absent |
| `normalised` | Normalisation discarded the physical scale, so the values are dimensionless | absent |

!!! warning "Normalised cutouts are dimensionless"
    Normalisation stretches pixels into the `data_type` range, so a normalised cutout is dimensionless whatever `apply_flux_conversion` says. To keep pixels in Jy, see [keep physical units](../how_to/keep_physical_units.md).

`BUNIT` **is** written when the resize did not conserve flux: the unit is still Jy, and `FLUXAPPX = T` carries the caveat that the values only approximate it.
