[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Cutana UI

Interactive user interface for creating astronomical cutouts from large FITS image datasets.
For a demonstration, see the [Cutana UI Demo Notebook](../examples/cutana_ui_demo.ipynb)

## Quick Start Guide

The Cutana UI guides you through a simple 3-step process to generate cutouts:

**Step 1**: Select your source catalogue (CSV file) containing astronomical object coordinates  
**Step 2**: Configure which image extensions to use and output settings  
**Step 3**: Start processing and monitor progress with live preview and status updates

### Launch the Interface
```python
import cutana_ui
cutana_ui.start(ui_scale=0.75)
```
You can vary the UI scale to fit your screen if necessary.
Recommended Range = 0.7-0.9 (max range 0.6-1.0)

## How It Works

### Start Screen: Setup Your Data
- **Select Catalogue**: Choose a CSV file containing your astronomical sources (see the input requirements below)
- **Choose Output Folder**: Set where your cutouts will be saved
- **Configure Extensions**: Select which FITS image extensions to process
- **Set Resolution**: Choose output image size (default: 256×256 pixels)
- **Select Output Format**: FITS files (one per source) or ZARR format (efficient for large datasets)

### Main Screen: Process & Monitor
Once configured, the main screen provides three panels:

- **Configuration Panel** (left): Adjust processing settings including:
  - Number of output channels (e.g., 3 for RGB images)
  - Channel combination weights (how to blend different extensions)
  - Normalisation method for consistent image appearance (powered by [fitsbolt](https://github.com/Lasloruhberg/fitsbolt))
  - Interpolation order for the resizing
  - Flux conserved resizing: Output is locked to float32 and none normalisation. resizing will be done with drizzle(slow)


- **Preview Panel** (top right): Shows sample cutouts from your first image tile so you can verify settings before processing all data. When clicking on the "reload" button in the top right, a new set of targets will be displayed.
The images will be sampled from the upper quartile of objects by size from up to 20k random samples from the provided catalogue.

- **Status Panel** (bottom right): Real-time progress tracking with logs and the ability to start/stop processing. The targets are sorted by tile before starting the cutout process, which might take some time.

## Input Data Requirements

Your source catalogue must be a CSV/FITS/PARQUET file with these columns (it must contain either diameter in arcsec or in pixels):

- `SourceID`: Unique identifier for each astronomical object
- `RA`: Right Ascension in degrees (0-360°, ICRS coordinate system)
- `Dec`: Declination in degrees (-90 to +90°, ICRS coordinate system)
- `diameter_arcsec`: Source size in arcseconds (must be > 0)
- `diameter_pixel`: Source size in pixels (must be > 0) 
- `fits_file_paths`: Path(s) to FITS files containing the source (as Python list format)

**Example CSV:**
```csv
SourceID,RA,Dec,diameter_arcsec,diameter_pixel,fits_file_paths
GaiaDR3_001,150.1234,2.5678,10.0,256,"['/data/euclid/tile_001.fits']"
GaiaDR3_002,150.2468,2.4567,8.5,220,"['/data/euclid/tile_002.fits']"
```

## Output Formats

**FITS Format**: Each source creates a separate FITS file with each selected image extension as a separate HDU (Header Data Unit).

**ZARR Format**: All cutouts stored in a single, efficient hierarchical data structure. Ideal for large datasets, faster I/O, and cloud storage. (Recommended)

---

## General Notes

### Log Level Selector
The header includes a log level dropdown (Debug, Info, Warning, Error) that controls the verbosity of logs displayed in the Jupyter notebook output. This affects only the console output - file logging is unaffected.

**Note**: Using "Debug" log level can generate excessive output which may negatively impact Jupyter notebook performance. For normal operation, "Warning" (default) or "Info" is recommended.

### Datalabs Usage
If using Cutana on Datalabs, currently you will need to reopen a tab with the Datakab containing the Notebook/Process every 48 hours to keep it running.

### Runtime on Q1
Testing on 15 million sources on Euclid Q1, a processing time of ~20 hours was achieved.
Be aware that large catalogues will require higher amounts of memory, which will be the main performance limit.
In this case the initial validation in the start screen should take up to ~2 minutes.
After starting the cutout process it might take up to ~30 minutes, till the cutout processes start. This is necessary for efficiency in the actual cutout generation and the necessary validation steps on very large catalogues.
Executing on multiple smaller catalogues, split by tile, might therefore be the most efficient way.

## For Developers

### Architecture Overview

The UI uses a modular widget-based architecture built on ipywidgets:

### Code Structure

The interface is completely decoupled from the backend processing engine:

```
cutana_ui/
├── app.py                     # Main application entry point
├── start_screen/              # Initial configuration interface
├── main_screen/               # Processing interface with 3 panels
│   ├── configuration_panel.py
│   ├── preview_panel.py
│   └── status_panel.py
├── widgets/                   # Reusable UI components
└── utils/                     # Backend communication & utilities
```

### Key Dependencies

- `ipywidgets`: Jupyter notebook widget framework
- `ipyfilechooser`: File/folder selection dialogs  
- `matplotlib`: Image rendering for previews
- `psutil`: System resource monitoring

### Configuration Management

Each processing session automatically generates a timestamped configuration file in your output directory:

```json
{
  "source_catalogue": "/path/to/catalogue.csv",
  "output_dir": "/path/to/output", 
  "selected_extensions": [
    {"name": "VIS", "ext": "IMAGE"},
    {"name": "NIR-H", "ext": "IMAGE"}
  ],
  "target_resolution": 256,
  "num_sources": 12451,
  "max_workers": 8,
  "timestamp": "20250815_143022"
}
```

### Testing

```bash
# Basic configuration tests
pytest tests/ui/test_config_only.py

# Full UI tests (requires conda environment)
conda activate cutana
pytest tests/ui/
```

### Backend Communication

The UI communicates with the processing backend through a clean interface:

- `check_source_catalogue()`: Validates catalogue format and extracts metadata
- `generate_previews()`: Generates sample cutouts for preview
- `start_processing()`: Initiates the full cutout generation pipeline

For complete implementation examples, see `examples/cutana_ui_demo.ipynb`.
For an example of an end to end test using the interface functions see e.g. [End to End Test](./../tests/cutana/e2e/test_e2e_channel_combinations.py).

