[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Changelog

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
