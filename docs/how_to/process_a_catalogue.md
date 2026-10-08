[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Process a full catalogue

Use the `Orchestrator` to turn a whole catalogue into cutouts on disk. It spreads the work over worker processes and sizes them to the memory available.

## Choose the right API

| Use case | API |
|---|---|
| Quick-look / previews (< 1000 sources) | [`create_cutouts_direct()`](direct_cutouts.md) |
| Full catalogue processing | `Orchestrator` (this page) |
| Streaming / ML pipeline integration | [`StreamingOrchestrator`](stream_cutouts.md) |

## Run from a catalogue file

Set `source_catalogue` and call `run()`, which loads the catalogue for you:

```python
from cutana import Orchestrator, get_default_config

config = get_default_config()
config.source_catalogue = "sources.csv"
config.output_dir = "cutouts_output/"

orchestrator = Orchestrator(config)
result = orchestrator.run()
```

## Run from a DataFrame

If the catalogue is already in memory, pass it to `start_processing()`:

```python
import pandas as pd
from cutana import Orchestrator, get_default_config

catalogue_df = pd.read_csv("sources.csv")

config = get_default_config()
config.output_dir = "cutouts_output/"
config.output_format = "zarr"
config.target_resolution = 256
config.selected_extensions = [
    {"name": "VIS", "ext": "PrimaryHDU"},
    {"name": "NIR-H", "ext": "PrimaryHDU"},
    {"name": "NIR-J", "ext": "PrimaryHDU"},
]
config.channel_weights = {
    "VIS": [1.0, 0.0, 0.5],
    "NIR-H": [0.0, 1.0, 0.3],
    "NIR-J": [0.0, 0.0, 0.8],
}

orchestrator = Orchestrator(config)
result = orchestrator.start_processing(catalogue_df)
```

Both calls return a dict with `status` (`"completed"`, `"failed"` or `"stopped"`), `total_sources`, `completed_batches`, `mapping_parquet` (the source-to-Zarr mapping) and, on failure, `error`. Check `status` before using the output: a run whose workers failed reports `"failed"` and lists them in `failed_processes`.

## Follow progress

```python
progress = orchestrator.get_progress()
print(f"Completed: {progress['completed_sources']}/{progress['total_sources']}")
```

`get_progress_for_ui()` returns a `ProgressReport` with the same numbers plus system memory, for display:

```python
report = orchestrator.get_progress_for_ui()
print(f"Progress: {report.progress_percent:.1f}%")
print(f"Memory: {report.memory_used_gb:.1f}/{report.memory_total_gb:.1f} GB")
```

## Stop or resume a run

```python
result = orchestrator.stop_processing()
print(f"Stopped {len(result['stopped_processes'])} processes")

if orchestrator.can_resume():
    print("Previous workflow can be resumed")
```

## Save and reuse a configuration

```python
from cutana import load_config_toml, save_config_toml

save_config_toml(config, "cutana_config.toml")
config = load_config_toml("cutana_config.toml")  # merged with the defaults
```

## Validate a catalogue before a long run

```python
from cutana.catalogue_preprocessor import CatalogueValidationError, load_and_validate_catalogue

try:
    catalogue_df = load_and_validate_catalogue("sources.csv")
    print(f"Loaded {len(catalogue_df)} sources")
except CatalogueValidationError as e:
    print(f"Validation error: {e}")
```

See [performance and memory](../explanation/performance.md) for how the worker count is chosen and how to keep a long run stable.
