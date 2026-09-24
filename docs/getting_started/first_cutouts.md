[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Your first cutouts

This tutorial takes you from a source catalogue to a folder of cutouts and back into Python. You need Cutana [installed](installation.md) and at least one FITS tile.

## 1. Write a catalogue

Create `sources.csv` with one row per source. `fits_file_paths` lists the FITS files that contain the source, one per band:

```csv
SourceID,RA,Dec,diameter_pixel,fits_file_paths
TILE_102018666_12345,45.123,12.456,128,"['/path/to/tile_vis.fits']"
TILE_102018666_12346,45.124,12.457,256,"['/path/to/tile_vis.fits']"
```

Every column is described in the [catalogue format reference](../reference/catalogue_format.md).

## 2a. Make cutouts in the UI

For most users, the interactive interface is the easiest way in. In a Jupyter notebook, run:

```python
import cutana_ui

cutana_ui.start()  # optionally set e.g. ui_scale=0.6 for a smaller UI
```

The interface walks you through three steps:

1. **Select your source catalogue** (`sources.csv`)
2. **Configure processing parameters** (image extensions, output format, resolution)
3. **Monitor progress** with live previews and status updates

## 2b. Or make cutouts from Python

For scripts and automated workflows, configure a run and hand it to the `Orchestrator`:

```python
from cutana import Orchestrator, get_default_config

config = get_default_config()
config.source_catalogue = "sources.csv"
config.output_dir = "cutouts_output/"
config.output_format = "zarr"  # or "fits"
config.target_resolution = 256
config.selected_extensions = [{"name": "VIS", "ext": "PrimaryHDU"}]
config.channel_weights = {"VIS": [1.0]}  # one output channel from VIS
config.console_log_level = "INFO"  # show progress in the console

orchestrator = Orchestrator(config)
result = orchestrator.run()
print(result["status"])
```

## 3. Look at the result

With Zarr output, `cutouts_output/` contains one folder per batch, each with an `images.zarr` archive and an `images_metadata.parquet` file that maps each image to its `SourceID`. Plot a few cutouts:

```python
import glob

import numpy as np
import zarr
from matplotlib import pyplot as plt

archive = glob.glob("cutouts_output/batch_*/images.zarr")[0]
images = zarr.open(archive, mode="r")["images"]  # shape (n_images, H, W, C)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax in axes.flatten():
    ax.imshow(images[np.random.randint(0, images.shape[0])], cmap="gray", origin="lower")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Next steps

- Mix several bands into colour images: [combine bands into channels](../how_to/combine_channels.md)
- Change the stretch: [normalise images](../how_to/normalise_images.md)
- Feed cutouts straight into a model: [stream cutouts](../how_to/stream_cutouts.md)
