[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Troubleshoot common errors

## `CatalogueValidationError: N catalogue rows are exact duplicates`

Cutout extraction is keyed by `SourceID`, so rows that share an ID *and* a position cannot be told apart and would collapse into one cutout. Deduplicate first:

```python
df = df.drop_duplicates(subset=["SourceID", "RA", "Dec"])
```

Sources that only share an ID at different positions are fine; Cutana separates them automatically. Unique IDs remain your responsibility above 100,000 rows, because Cutana never scans a whole catalogue and skips this check there in a single-shot load.

## A source is rejected for lying outside its FITS file

The catalogue check found a source whose centre falls outside a file its row names. Without the check it would become a cutout made entirely of edge padding. Fix the row's `fits_file_paths`, or set `skip_fits_check = True` if you have validated the catalogue yourself. See the [catalogue format reference](../reference/catalogue_format.md).

## `Streaming ended after N of M expected batches`

`get_batch_count()` is derived from the catalogue's row count, so the run expects one cutout per row, and fewer arrived. Call `get_delivery_report()` to see how many are missing and which worker they were assigned to. The usual causes are sources whose cutout window falls outside their tile (no cutout is produced, which is legitimate) and workers that died before delivering.

## `Worker <id> failed with exit code ...`

The message quotes the worker's own error. The full traceback is in the session log and in `logs/subprocesses/<id>_stderr.log`. Exit code `-9` means the kernel killed the worker for running out of memory: lower `N_batch_cutout_process` or `max_workers`.

## A run ends with fewer tiles than expected

A tile set that lacks the selected bands is skipped with `Skipping FITS set` in the log. Search the log for it before treating a short run as complete. See [band selection](../explanation/band_selection.md).

## Streaming hangs or leaks memory after an error

The batch loop was left without calling `cleanup()`. Wrap `init_streaming()` and the loop in `try: ... finally: orchestrator.cleanup()`, as shown in [stream cutouts](stream_cutouts.md).

## Problems on ESA Datalabs

Open a [service desk ticket](https://support.cosmos.esa.int/situ-service-desk/servicedesk/customer/portal/5).
