[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Combine bands into output channels

Cutana handles sources with several FITS files, one per band, and mixes them into output channels with configurable weights.

## 1. List the bands in the catalogue

Give each source every band file in `fits_file_paths`:

```csv
SourceID,RA,Dec,diameter_pixel,fits_file_paths
TILE_123_456,45.1,12.4,128,"['/path/to/vis.fits', '/path/to/nir_h.fits', '/path/to/nir_j.fits']"
```

Cutana identifies each file's band from its name, so `vis.fits` is `VIS`, `nir_h.fits` is `NIR-H`, and `nir_j.fits` is `NIR-J`.

## 2. Select the bands to load

`selected_extensions` decides which files of each set are loaded:

```python
config.selected_extensions = [
    {"name": "VIS", "ext": "PrimaryHDU"},
    {"name": "NIR-H", "ext": "PrimaryHDU"},
    {"name": "NIR-J", "ext": "PrimaryHDU"},
]
```

## 3. Map bands to output channels

`channel_weights` has one key per selected band. Each value lists that band's weight in each output channel, so three-element lists produce three channels, for example RGB:

```python
config.channel_weights = {
    "VIS": [1.0, 0.0, 0.5],
    "NIR-H": [0.0, 1.0, 0.3],
    "NIR-J": [0.0, 0.0, 0.8],
}
```

The keys are matched by band name, so dictionary order does not matter. Weights are not normalised.

To pick a band subset from a larger set, select only those bands and give `channel_weights` one entry per selected band. For the matching rules, what happens when a selection matches nothing, and how the UI discovers bands, see [band selection](../explanation/band_selection.md) and [channel mapping](../explanation/channel_mapping.md).
