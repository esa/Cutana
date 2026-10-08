[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Make cutouts in memory

For small requests (fewer than about 1,000 sources), `create_cutouts_direct()` runs entirely in your Python process and returns arrays. It avoids the orchestrator's worker processes, which makes it about 3x faster for quick looks and previews.

```python
import pandas as pd
from cutana import create_cutouts_direct, get_default_config

config = get_default_config()
config.target_resolution = 256
config.selected_extensions = [{"name": "VIS", "ext": "PrimaryHDU"}]
config.channel_weights = {"VIS": [1.0]}

catalogue_df = pd.read_csv("sources.csv")
results = create_cutouts_direct(catalogue_df, config)

for result in results:
    cutouts = result["cutouts"]  # ndarray (N, H, W, C)
    metadata = result["metadata"]  # list of per-source dicts
```

You get one result per FITS set. `selected_extensions` narrows which files of a set are loaded, as on every backend; see [band selection](../explanation/band_selection.md).

!!! warning "Memory"
    Every cutout stays in memory until you drop the results. At survey scale that adds up quickly, so use the [`Orchestrator`](process_a_catalogue.md) or [`StreamingOrchestrator`](stream_cutouts.md) for large catalogues.

## Process many tiles in parallel

When the sources span many FITS tiles, pass `max_workers` to process tiles concurrently on a thread pool. Each tile is loaded and closed in isolation.

```python
results = create_cutouts_direct(catalogue_df, config, max_workers=8)
```

`None` (the default) picks `min(n_tiles, effective_cpus, 8)` using the Kubernetes/cgroup-aware CPU count, and a single tile always runs serially. The cap of 8 is deliberate: the work is I/O bound, so beyond a handful of threads extra concurrency mostly oversubscribes a shared or networked filesystem instead of adding throughput.

## Extract once, render many times

For an interactive view where the user changes the stretch or colour mix, extract the raw arrays once and then combine and normalise them as often as needed:

```python
from cutana import apply_normalisation, combine_channels, create_cutouts_direct

config.do_only_cutout_extraction = True  # raw arrays: no resize, mix or stretch
results = create_cutouts_direct(catalogue_df, config)

raw = results[0]["cutouts"]  # (N, H, W, N_extensions)
names = results[0]["channel_names"]  # the extension order the weights bind to

config.channel_weights = {"VIS": [0.0, 0.0, 1.0], "NIR-H": [0.66, 0.0, 0.0]}
mixed = combine_channels(raw, config.channel_weights, names)  # -> (N, H, W, 3)
display = apply_normalisation(mixed, config)  # re-run per stretch change
```

!!! warning
    Pass `result["channel_names"]` to `combine_channels`. The names are required to resolve the weights; missing, duplicate or ambiguous mappings raise an error.
