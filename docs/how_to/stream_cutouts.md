[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Stream cutouts into a pipeline

Use `StreamingOrchestrator` to process a large catalogue batch by batch, for example to feed a machine-learning model. Background workers prepare the next batches while you work on the current one.

```python
from cutana import StreamingOrchestrator, get_default_config

config = get_default_config()
config.source_catalogue = "sources.csv"
config.output_dir = "streaming_output/"
config.target_resolution = 256
config.selected_extensions = [
    {"name": "VIS", "ext": "PrimaryHDU"},
    {"name": "NIR-H", "ext": "PrimaryHDU"},
]
config.channel_weights = {"VIS": [1.0, 0.0], "NIR-H": [0.0, 1.0]}

orchestrator = StreamingOrchestrator(config)
try:
    # Inside the try: init_streaming() pre-spawns workers one by one, and a failure
    # partway through leaves the spawned ones waiting until cleanup() releases them
    orchestrator.init_streaming(
        batch_size=10000,
        write_to_disk=False,  # return cutouts in memory, no disk I/O
    )
    for i in range(orchestrator.get_batch_count()):
        result = orchestrator.next_batch()
        # result["cutouts"]: list of numpy arrays, one per source (H, W, C)
        # result["metadata"]: list of source metadata dicts
        # result["batch_number"]: 1-indexed batch number
        process_cutouts(result["cutouts"])  # your inference or analysis
finally:
    orchestrator.cleanup()
```

!!! warning "Always call `cleanup()` in a `finally`"
    Workers block until they can hand over their next chunk. Leaving the loop early, including through an exception from `next_batch()`, strands them holding their shared-memory pools. Wrap `init_streaming()` and the batch loop in `try: ... finally: orchestrator.cleanup()`.

## `init_streaming()` parameters

| Parameter | Meaning |
| --- | --- |
| `batch_size` | Sources per batch. With `write_to_disk=False`, batches are cut to exactly this size. With `write_to_disk=True`, each batch is one Zarr archive written by one worker, so `batch_size` is a **minimum**: Cutana enlarges it to fit more sources into each worker, and `get_batch_count()` reports correspondingly fewer batches |
| `write_to_disk` | `False` returns cutouts through shared memory (recommended for ML pipelines); `True` writes each batch to a Zarr archive |
| `max_workers` | Maximum number of parallel workers |
| `min_workers` | Workers to start at init time, so they are already processing when `next_batch()` is first called |
| `max_shm_memory_consumption` | Total shared-memory budget in bytes |

## Check for missing cutouts

A source whose cutout window falls outside its tile produces no cutout, so a run can legitimately end with fewer cutouts than the catalogue has rows. `get_delivery_report()` tells you how many are missing and which worker they were assigned to:

```python
report = orchestrator.get_delivery_report()
if report["missing"]:
    print(f"{report['missing']} cutouts not produced: {report['shortfalls']}")
```

`shortfalls` is a list of `(process_id, assigned, delivered)` tuples.
