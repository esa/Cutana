[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Cutana Paper Benchmarking Suite

Comprehensive benchmarking scripts for the Cutana paper, comparing performance against naive Astropy baseline with proper HPC benchmarking practices.

## Quick Start

```bash
# Activate environment
conda activate cutana

# Test mode (fast, ~5 minutes)
cd paper_scripts
python create_results.py --test

# Full benchmarks with small catalogues (~2-4 hours)
python create_results.py --size small

# Full benchmarks with big catalogues (~4-8 hours)
python create_results.py --size big
```

## Folder Structure

```
paper_scripts/
├── catalogues/                  # Input catalogues
│   ├── small/                  # Smaller datasets for testing
│   │   ├── 50k-1tile-4channel.csv
│   │   ├── 1k-8tiles-4channel.csv
│   │   └── 50k-4tiles-1channel.csv
│   └── big/                    # Full-size datasets for paper
│       └── (same structure as small/)
├── data/                       # FITS files (symlink or copy tiles here)
├── results/                    # Raw benchmark data (JSON/CSV)
├── figures/                    # Plots for paper (PNG)
├── latex/                      # LaTeX macros for paper (TEX)
├── benchmark_config.toml       # Central configuration file
├── plots.py                    # Plotting functions module
├── astropy_baseline.py         # Enhanced Astropy baseline implementation
├── run_framework_comparison.py # Framework comparison benchmark
├── run_memory_profile.py       # Memory profiling benchmark
├── run_scaling_study.py        # Thread scaling study
├── create_results.py           # Master execution script
└── create_small_catalogues.py  # Script to generate smaller catalogues
```

## Benchmarks

### 1. Framework Comparison
Compares Astropy baseline (1 thread & 4 threads) vs Cutana (1 worker & 4 workers):
```bash
python run_framework_comparison.py --size small
python run_framework_comparison.py --size big
python run_framework_comparison.py --test  # Only 12k-1tile-4channel
```

**Scenarios:**
- 1 tile, 4 FITS, 50k sources
- 8 tiles, 4 FITS/tile, 1k sources (8k total)
- 4 tiles, 1 FITS/tile, 50k sources (12.5k per tile)

**Output:** `results/framework_comparison_*.json`, `results/framework_comparison_summary_*.csv`

### 2. Memory Profiling
Tracks memory usage over time for 1 tile scenario:
```bash
python run_memory_profile.py --size small
python run_memory_profile.py --test  # Use 12k-1tile-4channel
```

**Profiles:**
- Astropy baseline (4 threads) - best baseline performance
- Cutana 1 worker
- Cutana 4 workers

**Output:** `figures/memory_profile_*.png`, `results/memory_profile_stats_*.json`

### 3. Thread Scaling Study
Analyzes scaling from 1-8 workers (tests: 1, 2, 4, 6, 8):
```bash
python run_scaling_study.py --size small
python run_scaling_study.py --test  # Use 100k-1tile-4channel
```

**Metrics:**
- Runtime vs workers
- Throughput vs workers
- Speedup factor
- Parallel efficiency

**Output:** `figures/scaling_study_*.png`, `results/scaling_metrics_*.json`

### 4. LaTeX Values
Generates LaTeX macros from benchmark results:
```bash
python create_latex_values.py
```

**Output:** `latex/latex_values.tex`, `latex/benchmark_summary.txt`

## Configuration

### Configuration File: `benchmark_config.toml`

All benchmark parameters are now centrally configured in `benchmark_config.toml`:

```toml
[astropy_baseline]
target_resolution = 256                    # Target resolution for resizing (pixels)
apply_flux_conversion = true               # Enable flux conversion (AB magnitude)
interpolation = "bilinear"                 # Interpolation: nearest, bilinear, bicubic, biquadratic
zeropoint_keyword = "MAGZERO"             # FITS header keyword for AB zeropoint

[cutana]
target_resolution = 256                    # Target resolution for cutouts
N_batch_cutout_process = 1000             # Batch size for processing
output_format = "zarr"                     # Output format: zarr or fits
data_type = "uint8"                        # Data type: uint8, uint16, int16, float32, float64
normalisation_method = "none"              # Normalization method
interpolation = "bilinear"                 # Interpolation method
apply_flux_conversion = true               # Enable flux conversion

[framework_comparison]
warmup_cache = true                        # Warm up FITS cache before benchmarks
warmup_size = 1000                         # Number of sources for cache warmup

[plots]
dpi = 300                                  # Resolution for saved plots
figure_width = 12                          # Figure width in inches
figure_height = 6                          # Figure height in inches
```

**Edit this file to customize benchmark parameters without modifying code.**

## HPC Benchmarking Features

✅ **Cache warming** - Pre-loads FITS headers before measurements
✅ **Progress tracking** - Shows warmup progress every 10 sources, benchmark progress every 1000 sources
✅ **Memory management** - Explicitly closes files to avoid buildup
✅ **Realistic I/O** - Simulates HPC scenario with cached metadata
✅ **Multiple runs** - Scales tests across different worker counts
✅ **Detailed logging** - INFO level logs show real-time progress and statistics

## Catalogues

### Small (for testing/development)
- **50k-1tile-4channel**: 50k sources, 1 tile, 4 FITS files
- **1k-8tiles-4channel**: ~8k sources (1k/tile), 8 tiles, 4 FITS/tile
- **50k-4tiles-1channel**: 50k sources (~12.5k/tile), 4 tiles, 1 FITS/tile

### Big (for paper)
Same structure as small, larger source counts for final results.

## Expected Runtimes

| Mode | Time | Description |
|------|------|-------------|
| Test | ~3-5 min | Only 50k-1tile-4channel |
| Small | ~30-60 min | All 3 scenarios, small catalogues (50k sources each) |
| Big | ~2-4 hrs | All 3 scenarios, big catalogues (larger datasets) |

Individual scripts:
- Framework comparison: ~20-40 min (3 scenarios × 3 methods, 50k sources each)
- Memory profiling: ~10-15 min (1 scenario, 3 methods, 50k sources)
- Scaling study: ~30-45 min (1 scenario, 5 worker counts, 50k sources)
- LaTeX generation: <1 min

## For the Paper

After running benchmarks:

1. **Plots:** Copy from `figures/` folder:
   - `memory_profile_*.png`
   - `scaling_study_*.png`

2. **LaTeX macros:** Include `latex/latex_values.tex` in paper preamble:
   ```latex
   \input{path/to/latex_values.tex}
   ```

3. **Use in text:**
   ```latex
   Cutana achieves \cutanaFourRate{} cutouts per second with 4 workers,
   representing a \speedupFour{}× speedup over the Astropy baseline.
   ```

## Troubleshooting

**Missing catalogues:**
```
ERROR: Catalogue not found: paper_scripts/catalogues/small/100k-1tile-4channel.csv
```
→ Create catalogues or check path. Test catalogues should be in `catalogues/small/`.

**Memory errors:**
→ Use `--test` mode or smaller catalogues first.

**Long runtime:**
→ Use `--size small` for faster testing.

**FITS file paths:**
→ Ensure FITS files are in `paper_scripts/data/` or update paths in catalogues.

## Enhanced Baseline Implementation

### Astropy Baseline (`astropy_baseline.py`)

The enhanced Astropy baseline now includes a **complete processing pipeline** for fair comparison.

**Thread Configurations:**
- Benchmarks run with explicit 1-thread and 4-thread configurations for direct comparison
- Thread limits are set to match Cutana's per-process behavior
- Uses OMP_NUM_THREADS, MKL_NUM_THREADS, etc. to control threading in numpy/scipy operations

#### Processing Steps:
1. **FITS Loading**: Memory-mapped loading with file caching
2. **Cutout Extraction**: Using `astropy.nddata.Cutout2D`
3. **Resizing**: Target resolution scaling with skimage (same as Cutana)
4. **Flux Conversion**: AB magnitude to Jansky conversion (configurable)
5. **Normalization**: 0-1 range normalization
6. **FITS Writing**: Individual FITS files per cutout (cleaned up after benchmark)

#### Timing Breakdown:
All benchmarks now generate **detailed timing breakdown charts** showing:
- Time spent in each processing step
- Percentage of total time per step
- Comparison between Astropy baseline (1t, 4t) and Cutana (1w, 4w)

**Output charts:**
- `figures/astropy_1t_timing_*.png` - Astropy 1-thread baseline timing breakdowns
- `figures/astropy_4t_timing_*.png` - Astropy 4-thread baseline timing breakdowns
- `figures/cutana_1w_timing_*.png` - Cutana 1 worker timing breakdowns
- `figures/cutana_4w_timing_*.png` - Cutana 4 workers timing breakdowns

This represents a **realistic research workflow** with all necessary processing steps, making the comparison fair and comprehensive.

## Citation

If you use these benchmarks, please cite the Cutana paper (citation TBD).

## Support

For issues or questions, open an issue on the [Cutana GitHub repository](https://github.com/ESA-Datalabs/Cutana).
