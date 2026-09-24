[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Installation

Install Cutana from PyPI:

```bash
pip install cutana
```

## Development install

To work on Cutana itself, clone the repository and install every extra with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ESA/Cutana.git
cd Cutana
uv sync --all-extras
```

Or create the conda environment instead:

```bash
conda env create -f environment.yml
conda activate cutana
```

Next: [make your first cutouts](first_cutouts.md).
