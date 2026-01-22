[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Cutana - Astronomical Cutout Pipeline

[![PyPI version](https://img.shields.io/pypi/v/cutana.svg)](https://pypi.org/project/cutana/)
[![PyPI downloads](https://img.shields.io/pypi/dm/cutana.svg)](https://pypi.org/project/cutana/)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2511.04429-blue)](https://doi.org/10.48550/arXiv.2511.04429)
[![License](https://img.shields.io/badge/License-ESA--PL%20Permissive-green)](LICENSE.txt)
[![Python](https://img.shields.io/pypi/pyversions/cutana.svg)](https://pypi.org/project/cutana/)

**Cutana** is a high-performance Python pipeline for creating astronomical image cutouts from large FITS datasets. It provides both an interactive **Jupyter-based UI** and a **programmatic API** for efficient processing of astronomical survey data like ESA Euclid observations.

![Cutana Demo](assets/cutana_demo_2x.gif)

> **Note:** Cutana is currently optimised for **Euclid Q1/IDR1 data**. Some defaults and assumptions are Euclid-specific:
> - Flux conversion expects the `MAGZERO` header keyword (configurable via `config.flux_conversion_keywords.AB_zeropoint`)
> - Filter detection patterns are tuned for Euclid bands (VIS, NIR-Y, NIR-H, NIR-J)
> - FITS structure assumes one file per channel/filter
>
> For other surveys, you may need to adjust these settings or disable flux conversion (`config.apply_flux_conversion = False`).

## Support for Datalabs Users

For users experiencing problems with Cutana in the ESA Datalabs environment, please open a service desk ticket at: https://support.cosmos.esa.int/situ-service-desk/servicedesk/customer/portal/5

## Quick Start
### Installation

```bash
pip install cutana
```

Or for development:
```bash
# Create conda environment
conda env create -f environment.yml
conda activate cutana
```

### Interactive UI (Recommended)
For most users, the interactive interface provides the easiest way to process astronomical data:

```python
import cutana_ui
cutana_ui.start() # optionally can specify e.g. ui_scale=0.6 for smaller UI
```

This launches a step-by-step interface where you can:
1. **Select your source catalogue** (CSV format)
2. **Configure processing parameters** (image extensions, output format, resolution)  
3. **Monitor progress** with live previews and status updates

### Programmatic API
For automated workflows or integration into larger systems:

```python
import pandas as pd
from cutana import get_default_config, Orchestrator

# Configure processing
config = get_default_config()
config.source_catalogue = "sources.csv" # See format below
config.output_dir = "cutouts_output/"
config.output_format = "zarr"  # or "fits"
config.target_resolution = 256
config.selected_extensions = [{'name': 'VIS', 'ext': 'PrimaryHDU'}]  # Extensions to process
# 1 output channel for VIS, details explained below
config.channel_weights = {
        "VIS": [1.0],
    }
config.console_log_level = "INFO" # Show INFO logs in console

# Process cutouts
orchestrator = Orchestrator(config)
results = orchestrator.run()
```

## Input Data Format

Your source catalogue must be a CSV/FITS/PARQUET file containing these columns:

```csv
SourceID,RA,Dec,diameter_pixel,fits_file_paths
TILE_102018666_12345,45.123,12.456,128,"['/path/to/tile_vis.fits', '/path/to/tile_nir.fits']"
TILE_102018666_12346,45.124,12.457,256,"['/path/to/tile_vis.fits','/path/to/tile_nir.fits']"
```

**Required Columns:**
- `SourceID`: Unique identifier for each astronomical object
- `RA`: Right Ascension in degrees (0-360°, ICRS coordinate system)
- `Dec`: Declination in degrees (-90 to +90°, ICRS coordinate system)  
- `diameter_pixel`: Cutout size in pixels (creates square cutouts). Alternatively `diameter_arcsec`.
- `fits_file_paths`: JSON-formatted list of FITS file paths containing the source, use consistent order

## Output Formats

**ZARR Format** (recommended): All cutouts stored in a efficient archives, ideal for large datasets and analysis workflows. Cutana uses the [Zarr format](https://zarr.readthedocs.io/en/stable/) for high-performance storage and the [images_to_zarr](https://github.com/gomezzz/images_to_zarr/) library for conversion. (See the Output section below for sample code to access)

**FITS Format**: Individual FITS files per source, best for compatibility with existing astronomical software. Mandatory format for `do_only_cutout_extraction`, 
which skips all processing aside from the flux converison, which can be disabled.

## WCS (World Coordinate System) Handling

### FITS Output
FITS cutouts **preserve full WCS information** with accurate astrometric calibration:

- **Automatic pixel scale correction**: When cutouts are resized (e.g., from extracted size to `target_resolution`), the WCS pixel scale is automatically adjusted to maintain accurate coordinate transformations
- **Reference coordinate centering**: WCS reference pixel (`CRPIX`) is set to the cutout center, with reference coordinates (`CRVAL`) pointing to the source position
- **Format compatibility**: Supports CD matrix, CDELT, and PC+CDELT WCS formats from original FITS files
- **Sky area preservation**: Total sky coverage remains constant while pixel scale adjusts for resize operations

### Zarr Output
**Important**: Zarr archives **do not contain WCS information**. The WCS data is not recorded in the image metadata stored within the Zarr files. The central point and image size (in pixels or arcseconds, depending on what is provided) is recorded.

### Flux-Conserved Output

For scientific analysis requiring photometric accuracy, Cutana supports flux-conserving resizing:

```python
config.flux_conserved_resizing = True
config.data_type = "float32"
config.normalisation_method = "none"
```

**Important**: When using flux-conserved resizing, you should:
- Set `data_type = "float32"` to preserve numerical precision
- Use `normalisation_method = "none"` to keep original flux values
- This mode preserves total flux during the resizing operation, essential for photometric measurements

Note: Flux conservation uses [drizzle](https://github.com/spacetelescope/drizzle), which may affect performance

## Multi-Channel Processing

Cutana automatically handles sources with multiple FITS files and allows channel mixing through configurable weights.

### Channel Weights Configuration

The `channel_weights` parameter controls how multiple FITS files are combined into output channels. Each key represents a FITS extension name, and the corresponding value is a list of weights for image channels respectively.

**Important**: The FITS files listed in your source catalogue's `fits_file_paths` column must be ordered to match the extensions defined in `channel_weights`. The weights are not normalised, so careful consideration should be given when choosing them.

```python
# Configure channel weights (ordered dictionary format)
config.channel_weights = {
    "VIS": [1.0, 0.0, 0.5],    # RGB weights for VIS band
    "NIR-H": [0.0, 1.0, 0.3],  # RGB weights for NIR H-band
    "NIR-J": [0.0, 0.0, 0.8]   # RGB weights for NIR J-band
}
```

**Example**: If your source catalogue contains:
```csv
SourceID,RA,Dec,diameter_pixel,fits_file_paths
TILE_123_456,45.1,12.4,128,"['/path/to/vis.fits', '/path/to/nir_h.fits', '/path/to/nir_j.fits']"
```

The order of files in `fits_file_paths` must correspond to the order of keys in `channel_weights`:
1. `/path/to/vis.fits` → `VIS` extension
2. `/path/to/nir-h.fits` → `NIR-H` extension
3. `/path/to/nir-j.fits` → `NIR-J` extension

---

### Image Normalisation

Cutana supports multiple stretch algorithms with unified parameter configuration:

```python
config = get_default_config()

# Set normalisation method
config.normalisation_method = "asinh"  # "linear", "log", "asinh", "zscale"

# Configure normalisation parameters (method-specific defaults applied automatically)
config.normalisation.percentile = 99.8  # Data clipping percentile
config.normalisation.a = 0.7            # Transition parameter (asinh/log)
config.normalisation.n_samples = 1000   # ZScale samples  
config.normalisation.contrast = 0.25    # ZScale contrast
```

**Image stretching is powered by [fitsbolt](https://github.com/Lasloruhberg/fitsbolt) for consistent processing.**

config_path = save_config_toml(config, f"{config.output_dir}/cutana_config.toml")

### Output
In the case of zarr files the output will be organised in batches.
Per batch one folder is created each with an `images.zarr` and an `images_metadata.parquet`.
The folders are named using the format `batch_cutout_process_{index}_{unique_id}` where each process gets a unique identifier.

Setting `do_only_cutout_extraction` to True along with output_format `fits` allows cutouts to be directly created without normalisation/resizing or channel combination.
This is currently incompatible with zarr output. The flux conversion will still be applied and the `data_type` determined by the input.


#### Metadata
With the output zarr files, a metadata parquet file is created containing the following information: 
`source_id`, `ra`, `dec`, `idx_in_zarr`, `diameter_arcsec`, `diameter_pixel`, `processing_timestap`.

This can be read with:
```
import pandas as pd

metadata=pd.read_parquet("output_path/batch_cutout_process_*/images_metadata.parquet", engine='pyarrow')
```
This parquet provides a direct mapping between the individual images within the .zarr files and the Source
IDs and the cutouts. Note: No source IDs will be stored in the .zarr files!

`idx_in_zarr` is the index position of the source cutout within the .zarr file.

#### Images 
To open the .zarr files and look at example images, the following code can be used:

```
# Open the images
import zarr

# File: a dictionary that contains "images" of shape n_images,H,W,C
file = zarr.open("output_path/batch_cutout_process_*/images.zarr", mode='r')
print(list(file.keys()))

# Example plots
from matplotlib import pyplot as plt
import numpy as np

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for ax in axes.flatten():
    index=np.random.randint(0, file['images'].shape[0])
    image = file['images'][index]
    ax.imshow(image, cmap='gray',origin="lower")
    ax.axis('off')
plt.tight_layout()
plt.show()
```
If fits was selected as the output, then the fits files contain the info in the header of the `PRIMARY` extension.

To select a specific image from the .zarr files, the above output parquet file must be used as described in the 
Metadata section.

## Cutout Extraction Behavior

### Padding Factor

The `padding_factor` (in UI `zoom-out`) parameter controls the extraction area relative to the **source size** from your catalogue:

- **`padding_factor = 1.0`** (default): Extracts exactly the source size (`diameter_pixel` or `diameter_arcsec`)
- **`padding_factor < 1.0`** (zoom-in): Extracts a smaller area (source_size × padding_factor pixels)
  - Minimum value: 0.25 (extracts 1/4 of source_size)
  - Example: `padding_factor = 0.5` with 10px source extracts 5×5 pixels
- **`padding_factor > 1.0`** (zoom-out): Extracts a larger area (source_size × padding_factor pixels)
  - Maximum value: 10.0 (extracts 10× source_size)
  - Example: `padding_factor = 2.0` with 10px source extracts 20×20 pixels

All extracted cutouts are then resized to the final `target_resolution` (e.g., 256×256) in the processing pipeline.


## Stretch/Normalisation Configuration

Cutana supports multiple image stretching methods with **unified parameter naming** for optimal visualisation and analysis. The `normalisation_method` parameter controls the stretch algorithm, with a single set of parameters that automatically apply appropriate defaults based on the chosen method:

**Unified Parameter Details:**
All normalisation parameters are now stored in the `config.normalisation` DotMap object:

- **`config.normalisation.percentile`**: Percentile for data clipping, applied to all stretch methods (default: 99.8, range: 0-100)
- **`config.normalisation.a`**: Unified transition parameter with method-specific defaults:
  - ASINH: 0.7 (controls linear-to-logarithmic transition, range: 0.001-3.0)
  - Log: 1000.0 (scale factor for transition point, range: 0.01-10000.0)
- **`config.normalisation.n_samples`**: Number of samples for ZScale algorithm (default: 1000, range: 100-10000)
- **`config.normalisation.contrast`**: Contrast adjustment for ZScale (default: 0.25, range: 0.01-1.0)

**Note**: The unified `a` parameter automatically applies the appropriate default value based on the selected normalisation method, eliminating the need for method-specific parameter names.

### Linear Stretch
```python
from cutana import get_default_config

config = get_default_config()
config.normalisation_method = "linear"
config.normalisation.percentile = 99.8             # Percentile clipping (default)
```

### ASINH Stretch (Recommended)
```python
from cutana import get_default_config

config = get_default_config()
config.normalisation_method = 'asinh'
config.normalisation.percentile = 99.8             # Percentile clipping (default)
config.normalisation.a = 0.7                       # Transition parameter (default for asinh)
```

### Log Stretch
```python
from cutana import get_default_config

config = get_default_config()
config.normalisation_method = 'log'
config.normalisation.percentile = 99.8             # Percentile clipping (default)
config.normalisation.a = 1000.0                    # Scale factor (default for log)
```

### ZScale Stretch
```python
from cutana import get_default_config

config = get_default_config()
config.normalisation_method = 'zscale'
config.normalisation.percentile = 99.8             # Percentile clipping (default)
config.normalisation.n_samples = 1000              # Number of samples (default)
config.normalisation.contrast = 0.25               # Contrast parameter (default)
```

## Performance Considerations

### Memory and CPU Optimization

Cutana is designed to optimally utilize available system memory and CPU cores while maintaining stability and preventing system overload. The pipeline includes intelligent load balancing that automatically adjusts worker processes and memory allocation based on real-time system monitoring.

### Memory Constraints

In most deployment scenarios, **memory will be the limiting factor** rather than CPU cores. Cutana's load balancer continuously monitors memory usage and dynamically adjusts the number of worker processes to prevent memory exhaustion.

### Critical Usage Guidelines

**IMPORTANT**: Users SHALL NOT run other memory-intensive activities in parallel with Cutana processing.

Running competing memory-intensive processes will interfere with Cutana's load balancing algorithms and may lead to:
- System crashes due to memory exhaustion
- Degraded performance and increased processing time
- Inconsistent or failed processing results
- Potential data corruption in extreme cases

### Optimal Usage

For best performance:
1. **Dedicated Resources**: Run Cutana on dedicated compute resources when possible
2. **Monitor System**: Use system monitoring tools to track memory usage during processing
3. **Batch Size Tuning**: Adjust `N_batch_cutout_process` based on available memory
4. **Worker Configuration**: Let the load balancer automatically determine optimal worker count, or manually set `max_workers` conservatively
   
The load balancer will automatically scale down processing if memory pressure increases, but prevention through proper resource management is always preferable.

## Advanced features

To avoid bright objects at the border of an cutout making the target too faint, 
the user can set
```
int x # larger than 1, smaller than config.target_resolution
config.normalisation.crop_enable=True
config.normalisation.crop_height = x
config.normalisation.crop_width = x
```
Then after resizing, during normalisation the image is cropped to crop_height x crop_width around the center and the maximum value is determined inside this cropped region.

## Configuration Parameters

The following table describes all configuration parameters available in Cutana:

| Parameter                                     | Type     | Default                   | Range/Allowed Values                         | Description                                             |
| --------------------------------------------- | -------- | ------------------------- | -------------------------------------------- | ------------------------------------------------------- |
| **General Settings**                          |
| `name`                                        | str      | "cutana_run"              | -                                            | Run identifier                                          |
| `log_level`                                   | str      | "INFO"                    | DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE | Logging level for files                                 |
| `console_log_level`                           | str      | "WARNING"                 | DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE | Console/notebook logging level                          |
| **Input/Output Configuration**                |
| `source_catalogue`                            | str      | None                      | File path                                    | Path to source catalogue CSV file (required)            |
| `output_dir`                                  | str      | "cutana_output"           | Directory path                               | Output directory for results                            |
| `output_format`                               | str      | "zarr"                    | zarr, fits                                   | Output format                                           |
| `data_type`                                   | str      | "float32"                 | float32, uint8                               | Output data type                                        |
| `flux_conserved_resizing`                     | bool     | False                     | -                                            | Enable flux-conserving resizing (use with float32 + none normalisation, uses drizzle (slower)) |
| **Processing Configuration**                  |
| `max_workers`                                 | int      | 16                        | 1-1024                                       | Maximum number of worker processes                      |
| `N_batch_cutout_process`                      | int      | 1000                      | 10-10000                                     | Batch size within each process                          |
| `max_workflow_time_seconds`                   | int      | 1354571                   | 600-5000000                                  | Maximum total workflow time (~2 weeks default)          |
| **Cutout Processing Parameters**              |
| `do_only_cutout_extraction`                   | bool     | False                     | -                                            | If True, must set "fits", no norm/resize/combination img|
| `target_resolution`                           | int      | 256                       | 16-2048                                      | Target cutout size in pixels (square cutouts)           |
| `padding_factor`                              | float    | 1.0                       | 0.25-10.0                                    | Padding factor for cutout extraction (1.0 = no padding) |
| `normalisation_method`                        | str      | "linear"                  | linear, log, asinh, zscale, none             | Normalisation method, method must not be none for unit8 output |
| `interpolation`                               | str      | "bilinear"                | bilinear, nearest, cubic, lanczos            | Interpolation method                                    |
| **FITS File Handling**                        |
| `fits_extensions`                             | list     | ["PRIMARY"]               | List of str/int                              | Default FITS extensions to process                      |
| `selected_extensions`                         | list     | []                        | List of str/int/dict                         | Extensions selected by user (set by UI)                 |
| `available_extensions`                        | list     | []                        | List                                         | Available extensions (discovered during analysis)       |
| **Flux Conversion Settings**                  |
| `apply_flux_conversion`                       | bool     | True                      | -                                            | Whether to apply flux conversion (for Euclid data)      |
| `flux_conversion_keywords.AB_zeropoint`       | str      | "MAGZERO"                 | -                                            | Header keyword for AB magnitude zeropoint               |
| `user_flux_conversion_function`               | callable | None                      | -                                            | Custom flux conversion function (deprecated)            |
| **Image Normalization Parameters**            |
| `normalisation.percentile`                    | float    | 99.8                      | 0-100                                        | Percentile for data clipping                            |
| `normalisation.a`                             | float    | 0.7 (asinh), 1000.0 (log) | 0.001-10000.0                                | Unified transition parameter                            |
| `normalisation.n_samples`                     | int      | 1000                      | 100-10000                                    | Number of samples for ZScale algorithm                  |
| `normalisation.contrast`                      | float    | 0.25                      | 0.01-1.0                                     | Contrast adjustment for ZScale                          |
| `normalisation.crop_enable`                   | bool     | False                     | -                                            | Enable cropping during normalization                    |
| `normalisation.crop_width`                    | int      | -                         | 0-5000                                       | Crop width in pixels                                    |
| `normalisation.crop_height`                   | int      | -                         | 0-5000                                       | Crop height in pixels                                   |
| **Advanced Processing Settings**              |
| `channel_weights`                             | dict     | {"PRIMARY": [1.0]}        | Dict of str: list[float]                     | Channel weights for multi-channel processing            |
| `external_fitsbolt_cfg`                       | DotMap   | None                      | FITSBolt config or None                      | External FITSBolt config for ML pipeline integration (overrides normalisation settings) |
| **File Management**                           |
| `tracking_file`                               | str      | "workflow_tracking.json"  | -                                            | Job tracking file                                       |
| `config_file`                                 | str      | None                      | File path                                    | Path to saved configuration file                        |
| **Load Balancer Configuration**               |
| `loadbalancer.memory_safety_margin`           | float    | 0.15                      | 0.01-0.5                                     | Safety margin for memory allocation (15%)               |
| `loadbalancer.memory_poll_interval`           | int      | 3                         | 1-60                                         | Poll memory every N seconds                             |
| `loadbalancer.memory_peak_window`             | int      | 30                        | 10-300                                       | Track peak memory over N second windows                 |
| `loadbalancer.main_process_memory_reserve_gb` | float    | 4.0                       | 0.5-10.0                                     | Reserved memory for main process                        |
| `loadbalancer.initial_workers`                | int      | 1                         | 1-8                                          | Start with N workers until memory usage is known        |
| `loadbalancer.max_sources_per_process`        | int      | 150000                    | 1+                                           | Maximum sources per job/process                         |
| `loadbalancer.log_interval`                   | int      | 30                        | 5-300                                        | Log memory estimates every N seconds                    |
| `loadbalancer.event_log_file`                 | str      | None                      | File path                                    | Optional file path for LoadBalancer event logging       |
| `loadbalancer.skip_memory_calibration_wait`   | bool     | False                     | -                                            | Skip waiting for first worker memory measurements on launch of cutout creation and proceed immediately with a static memory estimate       |
| `process_threads`                             | int      | None                      | 1-128, None                                  | Thread limit per process (None = auto: cores // 4)      |
| **UI Configuration**                          |
| `ui.preview_samples`                          | int      | 10                        | 1-50                                         | Number of preview samples to generate                   |
| `ui.preview_size`                             | int      | 256                       | 16-512                                       | Size of preview cutouts                                 |
| `ui.auto_regenerate_preview`                  | bool     | True                      | -                                            | Auto-regenerate preview on config change                |

## Backend API Reference

### Orchestrator Class

The `Orchestrator` class is the main entry point for programmatic access to Cutana's cutout processing capabilities.

#### Constructor

```python
from cutana import Orchestrator
from cutana import get_default_config

orchestrator = Orchestrator(config, status_panel=None)
```

**Parameters:**
- `config` (DotMap): Configuration object created with `get_default_config()`
- `status_panel` (optional): UI status panel reference for direct updates

#### Main Processing Methods

##### `start_processing(catalogue_data)`

Start the main cutout processing workflow.

```python
import pandas as pd
from cutana import Orchestrator, get_default_config

# Load your source catalogue
catalogue_df = pd.read_csv("sources.csv")

# Configure processing
config = get_default_config()
config.output_dir = "cutouts_output/"
config.output_format = "zarr"
config.target_resolution = 256
config.selected_extensions = [{'name': 'VIS', 'ext': 'PrimaryHDU'}, {'name': 'NIR-H', 'ext': 'PrimaryHDU'},{'name': 'NIR-J', 'ext': 'PrimaryHDU'}]
config.channel_weights = {
    "VIS": [1.0, 0.0, 0.5],
    "NIR-H": [0.0, 1.0, 0.3],
    "NIR-J": [0.0, 0.0, 0.8]
}

# Process cutouts
orchestrator = Orchestrator(config)
result = orchestrator.start_processing(catalogue_df)
```

**Parameters:**
- `catalogue_data` (pandas.DataFrame): DataFrame containing source catalogue with required columns

**Returns:**
- `dict`: Result dictionary containing:
  - `status` (str): "completed", "failed", or "stopped"
  - `total_sources` (int): Number of sources processed
  - `completed_batches` (int): Number of completed processing batches
  - `mapping_parquet` (str): Path to source-to-zarr mapping `*.parquet` file
  - `error` (str): Error message if status is "failed"

##### `run()`

Simplified entry point that loads catalogue from config and runs processing.

```python
config = get_default_config()
config.source_catalogue = "sources.csv"
config.output_dir = "cutouts_output/"

orchestrator = Orchestrator(config)
result = orchestrator.run()
```

**Returns:**
- `dict`: Same format as `start_processing()`

#### Streaming Mode (Advanced)

Process large catalogues in batches for integration into data pipelines using `StreamingOrchestrator`.

The `StreamingOrchestrator` class provides a dedicated API for streaming workflows with optional **asynchronous batch preparation**, allowing the next batch to be prepared in the background while you process the current one.

```python
from cutana import get_default_config, StreamingOrchestrator

config = get_default_config()
config.source_catalogue = "sources.csv"
config.output_dir = "streaming_output/"
config.target_resolution = 256
config.selected_extensions = [{'name': 'VIS', 'ext': 'PrimaryHDU'}, {'name': 'NIR-H', 'ext': 'PrimaryHDU'}]
config.channel_weights =  {"VIS": [1.0,0.0],
                         "NIR-H": [0.0,1.0]}

# Create streaming orchestrator
orchestrator = StreamingOrchestrator(config)

# Initialize streaming - set synchronised_loading=False for async batch preparation
orchestrator.init_streaming(
    batch_size=10000,
    write_to_disk=False,  # Return cutouts in memory (zero disk I/O)
    synchronised_loading=False  # Prepare next batch in background
)

# Process batches
for i in range(orchestrator.get_batch_count()):
    result = orchestrator.next_batch()

    # result['cutouts']: numpy array of shape (N, H, W, C)
    # result['metadata']: list of source metadata dicts
    # result['batch_number']: 1-indexed batch number

    # Your ML inference or analysis here...
    process_cutouts(result['cutouts'])

    # With async mode, the next batch is already preparing in background!

orchestrator.cleanup()
```

**Key Parameters for `init_streaming()`:**
- `batch_size` (int): Maximum sources per batch
- `write_to_disk` (bool): If False, returns cutouts via shared memory (recommended for ML pipelines)
- `synchronised_loading` (bool):
  - `True` (default): Each batch is prepared when `next_batch()` is called
  - `False`: Next batch is prepared in background while you process the current one

**Async Mode Benefits:**
When `synchronised_loading=False`, Cutana spawns a subprocess to prepare the next batch while your code processes the current batch. If your processing time is similar to batch preparation time, you can achieve up to 2x throughput.

See `examples/async_streaming.py` for a benchmark comparing synchronous vs asynchronous streaming.

#### Progress and Status Methods

##### `get_progress()`

Get current progress and status information.

```python
progress = orchestrator.get_progress()
print(f"Completed: {progress['completed_sources']}/{progress['total_sources']}")
```

**Returns:**
- `dict`: Progress information including completed/total sources, runtime, errors

##### `get_progress_for_ui(completed_sources=None)`

Get progress information optimized for UI display.

```python
from cutana.progress_report import ProgressReport

progress_report = orchestrator.get_progress_for_ui()
print(f"Progress: {progress_report.progress_percent:.1f}%")
print(f"Memory: {progress_report.memory_used_gb:.1f}/{progress_report.memory_total_gb:.1f} GB")
```

**Parameters:**
- `completed_sources` (int, optional): Override completed sources count

**Returns:**
- `ProgressReport`: Dataclass with UI-relevant progress information including system resources

#### Control Methods

##### `stop_processing()`

Stop all active subprocesses gracefully.

```python
result = orchestrator.stop_processing()
print(f"Stopped {len(result['stopped_processes'])} processes")
```

**Returns:**
- `dict`: Stop operation results with list of stopped process IDs

##### `can_resume()`

Check if a workflow can be resumed from saved state.

```python
if orchestrator.can_resume():
    print("Previous workflow can be resumed")
```

**Returns:**
- `bool`: True if resumption is possible

### Configuration Functions

#### `get_default_config()`

Get the default configuration object.

```python
from cutana import get_default_config

config = get_default_config()
config.output_dir = "my_cutouts/"
config.target_resolution = 512
```

**Returns:**
- `DotMap`: Configuration object with all default parameters

#### `save_config_toml(config, filepath)`

Save configuration to TOML file.

```python
from cutana import save_config_toml

config_path = save_config_toml(config, "cutana_config.toml")
```

**Parameters:**
- `config` (DotMap): Configuration to save
- `filepath` (str): Path to save TOML file

**Returns:**
- `str`: Path to saved file

#### `load_config_toml(filepath)`

Load configuration from TOML file.

```python
from cutana import load_config_toml

config = load_config_toml("cutana_config.toml")
```

**Parameters:**
- `filepath` (str): Path to TOML configuration file

**Returns:**
- `DotMap`: Loaded configuration merged with defaults

### Catalogue Functions

#### `load_and_validate_catalogue(filepath)`

Load and validate source catalogue from CSV file using
`catalogue_preprocessor`.

```python
from cutana.catalogue_preprocessor import load_and_validate_catalogue

try:
    catalogue_df = load_and_validate_catalogue("sources.csv")
    print(f"Loaded {len(catalogue_df)} sources")
except CatalogueValidationError as e:
    print(f"Validation error: {e}")
```

**Parameters:**
- `filepath` (str): Path to CSV catalogue file

**Returns:**
- `pandas.DataFrame`: Validated catalogue DataFrame

**Raises:**
- `CatalogueValidationError`: If catalogue format is invalid