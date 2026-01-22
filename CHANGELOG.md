[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Changelog

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
