[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Configuration

Get a configuration with `get_default_config()`, change what you need, and pass it to a backend. Every parameter is an attribute of the returned `DotMap`, for example `config.target_resolution` or `config.normalisation.percentile`.

```python
from cutana import get_default_config

config = get_default_config()
config.output_dir = "my_cutouts/"
config.target_resolution = 512
```

## General

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `name` | str | `"cutana_run"` | - | Run identifier |
| `log_level` | str | `"INFO"` | DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE | Logging level for files |
| `console_log_level` | str | `"WARNING"` | DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE | Console/notebook logging level |

## Input and output

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `source_catalogue` | str | `None` | File path | Path to the source catalogue (required) |
| `output_dir` | str | `"cutana_output"` | Directory path | Output directory |
| `output_format` | str | `"zarr"` | zarr, fits | Output format |
| `data_type` | str | `"float32"` | float32, uint8 | Output data type |
| `flux_conserved_resizing` | bool | `False` | - | Flux-conserving resizing; use with float32 and `normalisation_method="none"`. Uses drizzle (slower) |

## Preprocessing

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `skip_catalogue_validation` | bool | `False` | - | Skip catalogue validation during preprocessing |
| `skip_fits_check` | bool | `False` | - | Skip the validation checks that open FITS files: that each row's tiles exist and are readable, and that the source falls inside them. Much faster on a short run, at the cost of those failures going undetected |

## Processing

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `max_workers` | int | Effective CPU count | 1-1024 | Maximum number of worker processes. The default honours Kubernetes/cgroup CPU limits |
| `N_batch_cutout_process` | int | `1000` | 10-10000 | Batch size within each process |
| `max_workflow_time_seconds` | int | `1354571` | 600-5000000 | Maximum total workflow time (about two weeks) |
| `process_threads` | int | `None` | 1-128, None | Thread limit per process (`None` = cores // 4) |

## Cutout processing

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `do_only_cutout_extraction` | bool | `False` | - | Raw cutouts: no normalisation, resizing or channel combination. Requires `output_format="fits"` |
| `target_resolution` | int | `256` | 16-2048 | Output cutout size in pixels (square) |
| `padding_factor` | float | `1.0` | 0.25-10.0 | Extraction area relative to the source size; see [padding factor](#padding-factor) |
| `normalisation_method` | str | `"linear"` | linear, log, asinh, zscale, none | Normalisation method; must not be `none` for uint8 output |
| `interpolation` | str | `"bilinear"` | bilinear, nearest, bicubic, lanczos | Interpolation method |

## FITS file handling

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `fits_extensions` | list | `["PRIMARY"]` | List of str/int | Default FITS extensions to process |
| `selected_extensions` | list | `[]` | List of str/int/dict | Bands to load; also narrows which files of a set are loaded. See [band selection](../explanation/band_selection.md) |
| `available_extensions` | list | `[]` | List | Available extensions (discovered during analysis) |

## Flux conversion

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `apply_flux_conversion` | bool | `True` | - | Convert pixels to Jy (for Euclid data) |
| `flux_conversion_keywords.AB_zeropoint` | str | `"MAGZERO"` | - | Header keyword for the AB magnitude zeropoint |
| `user_flux_conversion_function` | callable | `None` | - | Custom flux conversion function (deprecated) |

## Normalisation

All normalisation parameters live in `config.normalisation`. The transition parameter `a` takes a method-specific default, so you don't need method-specific names.

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `normalisation.percentile` | float | `99.8` | 0-100 | Percentile for data clipping, used by every method |
| `normalisation.a` | float | `0.7` (asinh), `1000.0` (log) | 0.001-10000.0 | Transition parameter. asinh: linear-to-log transition (0.001-3.0); log: scale factor (0.01-10000.0) |
| `normalisation.n_samples` | int | `1000` | 100-10000 | Number of samples for ZScale |
| `normalisation.asinh_n_samples` | int | `None` | 100-1000000 | Pixels per channel sampled to estimate the asinh percentile bounds. `None` uses every pixel, which is exact. Setting it trades accuracy for speed |
| `normalisation.contrast` | float | `0.25` | 0.01-1.0 | Contrast for ZScale |
| `normalisation.crop_enable` | bool | `False` | - | Find the stretch maximum in a central crop only |
| `normalisation.crop_width` | int | - | 0-5000 | Crop width in pixels |
| `normalisation.crop_height` | int | - | 0-5000 | Crop height in pixels |

## Channels

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `channel_weights` | dict | `{"PRIMARY": [1.0]}` | Dict of str: list[float] | Per-band weights for each output channel; see [combine bands into channels](../how_to/combine_channels.md) |
| `external_fitsbolt_cfg` | DotMap | `None` | fitsbolt config or None | External fitsbolt config for ML pipeline integration; overrides the normalisation settings. Create it with `fitsbolt.create_config()` |

## Files

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `tracking_file` | str | `"workflow_tracking.json"` | - | Job tracking file |
| `config_file` | str | `None` | File path | Path to a saved configuration file |

## Load balancer

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `loadbalancer.memory_safety_margin` | float | `0.15` | 0.01-0.5 | Safety margin for memory allocation |
| `loadbalancer.memory_poll_interval` | int | `3` | 1-60 | Poll memory every N seconds |
| `loadbalancer.memory_peak_window` | int | `30` | 10-300 | Track peak memory over N-second windows |
| `loadbalancer.main_process_memory_reserve_gb` | float | `4.0` | 0.5-10.0 | Memory reserved for the main process |
| `loadbalancer.initial_workers` | int | `1` | 1-8 | Workers to start with until memory usage is known |
| `loadbalancer.max_sources_per_process` | int | `150000` | 1+ | Maximum sources per worker process |
| `loadbalancer.log_interval` | int | `30` | 5-300 | Log memory estimates every N seconds |
| `loadbalancer.event_log_file` | str | `None` | File path | Optional file for load balancer event logging |
| `loadbalancer.skip_memory_calibration_wait` | bool | `False` | - | Start immediately with a static memory estimate instead of waiting for the first worker's measurements |

## UI

| Parameter | Type | Default | Allowed values | Description |
| --- | --- | --- | --- | --- |
| `ui.preview_samples` | int | `10` | 1-50 | Number of preview samples |
| `ui.preview_size` | int | `256` | 16-512 | Size of preview cutouts |
| `ui.auto_regenerate_preview` | bool | `True` | - | Regenerate the preview when the config changes |

## Padding factor

`padding_factor` (`zoom-out` in the UI) sets the extraction area relative to the source size in the catalogue (`diameter_pixel` or `diameter_arcsec`):

| Value | Effect | Example with a 10 px source |
| --- | --- | --- |
| `1.0` (default) | Extracts exactly the source size | 10 × 10 px |
| `< 1.0`, minimum `0.25` | Zooms in: extracts `source_size × padding_factor` | `0.5` extracts 5 × 5 px |
| `> 1.0`, maximum `10.0` | Zooms out: extracts `source_size × padding_factor` | `2.0` extracts 20 × 20 px |

Every extracted cutout is then resized to `target_resolution`.

## Save and load

```python
from cutana import load_config_toml, save_config_toml

save_config_toml(config, "cutana_config.toml")  # returns the path
config = load_config_toml("cutana_config.toml")  # merged with the defaults
```
