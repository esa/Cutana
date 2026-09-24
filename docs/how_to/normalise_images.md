[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Normalise images

Choose a stretch with `normalisation_method` and tune it through `config.normalisation`. Parameters you don't set take the method's default. Stretching is done by [fitsbolt](https://github.com/Lasloruhberg/fitsbolt).

## ASINH (recommended)

```python
from cutana import get_default_config

config = get_default_config()
config.normalisation_method = "asinh"
config.normalisation.percentile = 99.8  # percentile clipping (default)
config.normalisation.a = 0.7  # linear-to-log transition (asinh default)
# config.normalisation.asinh_n_samples = 10000  # optional: subsample for speed, not exact
```

`asinh_n_samples` estimates the percentile bounds from a pixel subsample instead of every pixel. Leave it at `None` for exact output; setting it biases the bright tail and changes output values.

## Linear

```python
config.normalisation_method = "linear"
config.normalisation.percentile = 99.8
```

## Log

```python
config.normalisation_method = "log"
config.normalisation.percentile = 99.8
config.normalisation.a = 1000.0  # scale factor (log default)
```

## ZScale

```python
config.normalisation_method = "zscale"
config.normalisation.percentile = 99.8
config.normalisation.n_samples = 1000  # number of samples (default)
config.normalisation.contrast = 0.25  # contrast (default)
```

## Keep bright neighbours from washing out the target

A bright object near the edge of a cutout can set the stretch's maximum and leave the target faint. Crop the region used to find the maximum to the centre of the cutout:

```python
config.normalisation.crop_enable = True
config.normalisation.crop_height = 64  # larger than 1, smaller than target_resolution
config.normalisation.crop_width = 64
```

After resizing, normalisation takes the maximum value from the central `crop_height` × `crop_width` region only.

## Use an external fitsbolt configuration

To match an existing ML pipeline exactly, build a config with `fitsbolt.create_config()`, adjust it, and pass it as `config.external_fitsbolt_cfg`. It overrides the normalisation settings above.

All parameters and their ranges are in the [configuration reference](../reference/configuration.md#normalisation).
