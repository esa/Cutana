[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Changelog

## [Unreleased]

## [v0.4.0] – 2026-09-24

### Fixed
- **`channel_weights` were applied by dictionary order, not by channel**: a weights dict ordered differently from a row's `fits_file_paths` silently mixed the bands. Weights now resolve by channel name
- **Silent cutout loss**: rows with duplicate `SourceID` that cannot be told apart are now refused, and a slow streaming consumer no longer makes workers drop chunks after 60s
- **Failed workers reported as success**: a fatal worker error now fails the run, and `Orchestrator` returns `status: failed` with `failed_processes`
- **A `selected_extensions` that matches no FITS set** raises instead of finishing as an empty run
- **`create_cutouts_direct()` ignored `selected_extensions`**, loading every file in a set and pairing weights with the wrong bands
- **`skip_fits_check` had no effect**: it is now a validated config key, default `False`
- **The `max_workers` default ignored the Kubernetes CPU quota**, oversubscribing Datalab pods
- **Normalised cutouts were labelled `approx Jy`**: `UNIT` is now `normalised`, and a physical unit is claimed only when the pixel scale is preserved
- **UI**: asinh runs failed on a missing `asinh_n_samples`, and raw-cutout mode had its forced normalisation overwritten
- **Catalogue discovery**: rows may list their bands in any order, `-` and `_` are interchangeable in filter names, the mixed-resolution check compares every tile, and estimated source counts show as `~N`
- **Extraction-only output** raises instead of renaming bands to `channel_1…N`
- Invalid-cast warning when converting `diameter_arcsec` to pixels

### Added
- **`combine_channels()` and `apply_normalisation()`** are exported from `cutana`, so raw cutouts can be re-mixed and re-stretched without re-extraction
- **Source-in-product check**: a source outside its FITS tiles is an error, and a cutout crossing a tile edge is a warning. Large catalogues are sampled; `skip_fits_check` turns it off
- **`max_workers` on `create_cutouts_direct()`** processes tiles on a thread pool, with a per-FITS-set progress heartbeat
- **`BUNIT` and `FLUXAPPX` FITS keywords**, written only when the unit is known
- **`normalisation.asinh_n_samples`**: opt-in subsampling of the asinh bounds for speed. The default stays exact
- **Streaming profiler** (`benchmarking/profile_cutana.py`, `benchmark` extra) and per-worker stage timings via `StreamingOrchestrator.get_worker_info()`

### Changed
- **Streaming starts far fewer worker processes**: internal batches are sized by the load balancer instead of 1,000 sources each. The speedup over 0.3.2 depends on per-cutout cost: ~3x at 64 px from a single band (599k sources), and 1.6x in memory or 1.1x to disk at 192 px from 4 bands into 3 channels. With `write_to_disk=True`, `batch_size` is now a minimum
- **Breaking**: `combine_channels()` requires `channel_names`
- **Resizing uses OpenCV** (`INTER_AREA`, plus a new `lanczos` option); `scikit-image` is dropped
- **`fitsbolt>=0.3.1,<0.4`**, with float32 channel combination
- **`UNIT` on the primary HDU is deprecated** in favour of `BUNIT`
- **Preview sources** come from the first 10,000 catalogue rows
- **Streaming failures surface**: worker errors, failed spawns and delivery shortfalls are raised instead of swallowed
- **UI "Raw cutout" label** shows `[Jy]` only when flux conversion is on

### Testing
- **Release test suite**: a 12-configuration `--matrix` compared pixel by pixel against Git LFS ground truth, cutout WCS checked against the parent tiles, and throughput compared with the previous release

### Documentation
- **Documentation site** with a generated API reference, replacing the old `docs/` guides
- **`REVIEW.md`** with repo-specific review guidance

### Removed
- 4 obsolete benchmark scripts

## [v0.3.2] – 2026-07-06

### Fixed
- **FITS cutout WCS re-tangenting bug**: cutout WCS now reproduces the parent tile mapping exactly — `CRVAL`, `CTYPE` and the CD/PC orientation are inherited from the parent unchanged and only `CRPIX` is shifted to the extraction origin — instead of re-tangenting the projection at each source. The previous approach left the cutout frame rotated by the meridian convergence between tile centre and source, producing a positional error that was ~0 at the cutout centre and grew toward the edges (~1″ at a few-arcmin FOV for sources far from the tile centre, worse near high \|Dec\| / tile corners). Only the FITS WCS header was affected; pixel data and Zarr/streaming outputs were not

### Added
- **`UNIT` and `CONSVFLX` FITS header keywords** on individual cutout outputs: `UNIT` records the pixel unit (`OriginalUnit`, `Jy`, `approx Jy`, or `approx OriginalUnit`) depending on whether flux conversion and flux-conserved resizing were applied, and `CONSVFLX` records whether flux-conserved resizing was used

### Changed
- **UI "Raw cutout" checkbox** relabelled to "Raw cutout (in Jy):" to make the output unit explicit
- **`fitsbolt` pinned to `==0.3.0`** (from `==0.2.0`)

### Documentation
- **README**: added a Flux Conversion section (pixels converted to Jansky by default via the `MAGZERO` keyword, disable with `config.apply_flux_conversion = False`), documented the `UNIT` / `CONSVFLX` header keywords, and clarified the "Raw cutouts" terminology

## [v0.3.1] – 2026-06-04

### Fixed
- **Streaming in-memory batch repacking**: when returning cutouts in memory, results are now drained from a rolling pending buffer in exact `batch_size` chunks, so every batch but the last contains exactly `batch_size` cutouts and the tail batch is no longer silently dropped (no sources lost). Disk mode (1:1 internal-to-user mapping) and `get_batch_count()` are unchanged
- **Large parquet catalogues**: string columns are now read with `pa.large_string()` (64-bit offsets), avoiding the 2 GB per-column string limit when loading large catalogues

### Changed
- **Streaming pending buffer** backed by a `deque` for O(`batch_size`) emission

### Documentation
- **README `StreamingOrchestrator` section** updated to reflect the parallel API (`max_workers`/`min_workers`, configurable background workers); removed the obsolete `synchronised_loading` async description

## [v0.3.0] – 2026-05-12

### Added
- **Direct cutout API** (`create_cutouts_direct()`) for fast in-process cutout generation without the orchestrator/worker overhead (#293)
- **Parallel `StreamingOrchestrator`** with adaptive multi-worker support and per-worker SHM pools — measured 2.81x speedup with 4 workers (#306)
- **Browser test infrastructure** with Playwright + Voila, including a session-scoped Voila server, video-on-failure recording, and `scripts/validate_browser_testing.py` for interactive MCP-driven testing (#297)
- **Per-source pixelscale and tile metadata** written into cutout outputs to make downstream WCS/registration trivial (#321, closes #219, #206)
- **`skip_catalogue_validation`** configuration flag to bypass catalogue validation for trusted/pre-validated inputs (#285)
- **`min_workers`** parameter on `StreamingOrchestrator` to set a floor for the adaptive worker scheduler (#308)
- **Show Config button** in the UI header with JSON highlighting and clipboard copy (#322)
- **`PerformanceProfiler` enhancements**: configurable timing functions, additional statistics, simplified logging output (#305)
- **`PLC0415` ruff rule** to flag imports outside top level (#302)

### Changed
- **Linter/formatter migrated from flake8 + black to ruff** (`ruff check` + `ruff format`, line-length 100) (#294)
- **Build/CI migrated from conda/micromamba to uv** for dependency management and faster installs (#296)
- **SourceID deduplication** now reformats colliding IDs as `sourceid_ra_dec` instead of silently dropping rows; uniqueness asserted per-tile across 1–4 bands (#283, #285)
- **FITS metadata schema stabilized** with observability hooks (#321)
- **Test suite parametrized** (67 fewer test functions, 78% coverage vs. 71%) and parallelized with pytest-xdist (#300)
- **Silent exception fallbacks removed** from cutana — broken invariants now raise instead of returning placeholder values (#316/#318/#319)
- **`getattr` config fallbacks removed**; missing config keys now error loudly (#317/#318)
- **`fitsbolt` pinned to `==0.2.0`**; output dtype is propagated end-to-end through the fitsbolt config (#289)
- **FITS writer** no longer hard-fails on `tile=None`; degraded path is supported (#321)
- **Streaming SHM pool protocol** simplified and pool mode enforced in the orchestrator
- **Show Config / UI panel** refactored to separate start-screen vs. main-screen tests and improve JSON rendering (#322)

### Fixed
- **macOS shared memory** names truncated to fit the platform limit (#292)
- **SHM resource tracker warnings** fixed by unregistering before unlinking (#276)
- **Silent duplicate SourceID** now reformatted rather than dropped; covered by per-tile and multi-band tests (#283)
- **Cutana `data_type`** propagated to fitsbolt with validation enforcement (#289, #291)
- **Silent normalisation fallback** replaced with `RuntimeError` (#275)
- **Silent extension drop** in channel-order/extension validator (#324, closes #315)
- **Windows worker stdout** read via dedicated threads to prevent IPC blocking
- **Windows 1-worker path** avoids the threading fallback that triggered platform-specific bugs
- **`combine_channels`** preserves input dtype when calling fitsbolt
- **pandas 2.x compatibility**: switch to `is_string_dtype` in test helpers

### Performance
- **Band-selective FITS loading** — only the FITS files for requested bands are opened (#280)
- **Parallel `StreamingOrchestrator`** — 2.81x speedup with 4 workers (#306)
- **Subprocess IPC bottleneck** eliminated; coverage instrumentation no longer fights the worker hot path
- **E2E tests** reduced in resolution and gated behind `slow` marker; unit-test parallelism via xdist

### Removed
- **94 low-value tests** that exercised stdlib rather than cutana behavior
- **15 redundant E2E padding tests** (folded into unit tests)
- **`convert_data_type`** function and its tests (dead code)
- **flake8 + black** configuration and tooling (replaced by ruff)
- **conda/micromamba** dev-environment files (replaced by uv)

---

## [v0.2.1] – 2025-01-21

### Changed
- **Default max_workers** now uses available CPU count instead of hardcoded 16

### Fixed
- **Status panel worker display** now shows "16 workers" before processing starts instead of misleading "0/16 workers"
- **Help panel README handling** now uses `importlib.metadata` to load main README from package metadata in pip-installed environments

---

## [v0.2.0] – 2025-01-12

### Added
- **Streaming mode** with `StreamingOrchestrator` for in-memory cutout processing using shared memory, enabling direct processing without disk I/O
- **Flux-conserved resizing** using the drizzle algorithm to preserve photometric accuracy during image resampling
- **Parquet input support** allowing source catalogues to be provided in Parquet format in addition to CSV
- **Raw cutout extraction** (`cutout_only` mode) for outputting unprocessed cutouts directly from FITS tiles
- **External FITSBolt configuration** support with TOML serialization for seamless integration with FITSBolt pipelines
- **Log level selector** dropdown in the UI header for runtime log verbosity control
- **Vulture dead code detection** CI workflow to identify and prevent unused code accumulation
- **Ruff import sorting** CI check to enforce consistent import ordering across the codebase
- **Comprehensive benchmarking suite** (`paper_scripts/`) for performance evaluation and reproducibility of paper results
- **Async streaming example** (`examples/async_streaming.py`) demonstrating programmatic streaming mode usage

### Changed
- **Catalogue streaming architecture** with `CatalogueStreamer` enabling memory-efficient processing of catalogues with 10M+ sources through atomic tile batching
- **Default output folder** changed from `cutana/output` to `cutana_output` for cleaner project structure
- **Default resizing mode** changed to symmetric for more intuitive cutout dimensions
- **Logging configuration** now follows loguru best practices: disabled by default, users opt-in via `logger.enable("cutana")`
- **WCS handling** optimized for FITS output with correct pixel scale and WCS - no SIP distortions implemented
- **Documentation** updated for Euclid DR1 compatibility with improved README and markdown formatting
- **Source mapping output** now written as Parquet instead of CSV for better performance with large catalogues
- **Dependencies** updated: `fitsbolt>=0.1.6`, `images-to-zarr>=0.3.5`, added `drizzle>=2.0.1`, `scikit-image>=0.21`

### Fixed
- **WCS pixel offset** corrected 1-based indexing and half-pixel offset issues affecting cutout positioning
- **Subprocess logging** resolved ANSI escape codes and duplicate log folder creation
- **Windows compatibility** fixed temp file permission issues in streaming mode tests
- **Parquet file selection** in UI now properly filters and displays Parquet files
- **Flux conservation integration** properly applied in `cutout_process_utils.py`
- **Normalisation bypass** allowing `"none"` config value to skip normalisation entirely

### Performance
- **10x memory reduction** for large catalogue processing through true streaming implementation
- **WCS computation optimisation** reducing overhead for FITS output generation
- **Single-threaded FITSBolt** mode for improved stability in multi-process environments

### Removed
- **`JobCreator` class** and associated dead code identified through vulture static analysis
- **Obsolete example notebooks** (`Cutana_IDR1_Setup.ipynb`, `backend_demo.ipynb`) replaced with updated documentation

---
