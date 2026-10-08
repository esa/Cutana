[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Cutana

**Cutana** is a high-performance Python pipeline for creating astronomical image cutouts from large FITS datasets. It provides both an interactive **Jupyter-based UI** and a **programmatic API** for processing survey data such as ESA Euclid observations.

![Cutana demo](https://raw.githubusercontent.com/ESA/Cutana/main/assets/cutana_demo_2x.gif)

!!! note "Optimised for Euclid"
    Cutana is currently optimised for **Euclid Q1/IDR1 data**. Some defaults and assumptions are Euclid-specific:

    - Flux conversion expects the `MAGZERO` header keyword (configurable via `config.flux_conversion_keywords.AB_zeropoint`) to convert to Jy
    - Filter detection patterns are tuned for Euclid bands (VIS, NIR-Y, NIR-H, NIR-J)
    - FITS structure assumes one file per channel/filter

    For other surveys, you may need to adjust these settings or disable flux conversion (`config.apply_flux_conversion = False`).

## Where to start

| If you want to | Read |
| --- | --- |
| Install Cutana and make your first cutouts | [Getting started](getting_started/installation.md) |
| Do a specific task, such as streaming cutouts into a model or mixing bands into RGB | [How-to guides](how_to/process_a_catalogue.md) |
| Look up a catalogue column, config parameter or output format | [Reference](reference/catalogue_format.md) |
| Understand band matching, WCS handling or performance behaviour | [Explanation](explanation/channel_mapping.md) |
| Look up a function or class | [API reference](api/index.md) |

## Support

- **Source code and issues**: [ESA/Cutana on GitHub](https://github.com/ESA/Cutana)
- **ESA Datalabs users**: open a [service desk ticket](https://support.cosmos.esa.int/situ-service-desk/servicedesk/customer/portal/5)
- **Citation**: [arXiv:2511.04429](https://doi.org/10.48550/arXiv.2511.04429)
