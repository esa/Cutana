[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Catalogue format

The source catalogue is a CSV, FITS or Parquet file with one row per source:

```csv
SourceID,RA,Dec,diameter_pixel,fits_file_paths
TILE_102018666_12345,45.123,12.456,128,"['/path/to/tile_vis.fits', '/path/to/tile_nir.fits']"
TILE_102018666_12346,45.124,12.457,256,"['/path/to/tile_vis.fits','/path/to/tile_nir.fits']"
```

## Required columns

| Column | Type | Description |
| --- | --- | --- |
| `SourceID` | str | Unique identifier for each object. See [unique IDs](#unique-ids) |
| `RA` | float | Right Ascension in degrees (0 to 360, ICRS) |
| `Dec` | float | Declination in degrees (-90 to +90, ICRS) |
| `diameter_pixel` | int | Cutout size in pixels; cutouts are square. Use `diameter_arcsec` (float) instead to give the size in arcseconds |
| `fits_file_paths` | list of str | FITS files containing the source, one per band, as a JSON-formatted list. Use the same band order in every row |

## Unique IDs

Cutout extraction is keyed by `SourceID`, so rows that share one collapse into a single cutout. Keep `SourceID`s unique across all rows.

For catalogues with fewer than 100,000 sources, Cutana detects duplicates and reformats their IDs as `SourceID_RA_Dec`. Rows that duplicate both the ID *and* the position cannot be told apart and are rejected with `CatalogueValidationError`; deduplicate them first with `df.drop_duplicates(subset=["SourceID", "RA", "Dec"])`.

!!! warning "Large catalogues"
    Above 100,000 sources the check is skipped, because Cutana never scans a whole catalogue, so uniqueness is your responsibility. Streaming processes small internal batches, so the check still applies within each batch.

## Source-in-product check

Cutana checks that each source falls inside the FITS files its row names:

- A source whose **centre lies outside** a file it names is rejected with `CatalogueValidationError`. Without the check it would become a full-size cutout made entirely of edge padding.
- A centre **inside** the image with the cutout box **crossing an edge** is only a warning, since a partial cutout is usually what you want at a survey boundary.

Large catalogues are sampled rather than checked row by row: 1,000 rows, opening at most 100 distinct files. What was skipped is logged.

The check opens FITS files, which takes a few seconds on a networked filesystem regardless of catalogue size. Set `skip_fits_check = True` to skip it together with the check that each file exists and is readable.
