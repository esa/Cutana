[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Read Zarr outputs

Zarr output is written in batches: one folder per batch, each holding an `images.zarr` archive and an `images_metadata.parquet` file. The layout is described in the [output formats reference](../reference/output_formats.md).

## Find a source's cutout

The Zarr archives store no source IDs. Row *i* of the metadata describes image *i* of the archive in the same folder, so look up a `SourceID` by its row position:

```python
import pandas as pd

batch = "output_path/batch_cutout_process_000_abc123"
metadata = pd.read_parquet(f"{batch}/images_metadata.parquet")
index = metadata.index[metadata["source_id"] == "TILE_102018666_12345"][0]
```

## Load images

```python
import zarr

archive = zarr.open(f"{batch}/images.zarr", mode="r")
images = archive["images"]  # shape (n_images, H, W, C)
cutout = images[index]
```

## Plot a sample

```python
import numpy as np
from matplotlib import pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax in axes.flatten():
    ax.imshow(images[np.random.randint(0, images.shape[0])], cmap="gray", origin="lower")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

With FITS output, each source gets its own file, and the information is in the header of the `PRIMARY` extension.
