[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Band selection

`selected_extensions` decides **which files of a FITS set are loaded**, on every backend. Names are matched against the band in each filename (`VIS`, `NIR-H`, …), so selecting three bands from a four-file Euclid set loads exactly those three, in catalogue order, and `channel_weights` needs one entry per selected band. Use `"PRIMARY"` to switch band selection off and load every file in the set.

## A selection that names no known band

A selection naming no band Cutana recognises narrows nothing, so every file in each set loads. On non-Euclid data that is the normal case: the UI labels a file the recogniser cannot classify `UNKNOWN`, and there is no band behind that label to select on.

This is logged once per run, because the other way to reach it is a typo: `"NIRH"` narrows nothing just as quietly.

## A selection that matches no file

!!! warning
    A selection that names real bands but matches no file in a set asks for data that set does not contain.

    - `create_cutouts_direct()` raises `ValueError`.
    - The orchestrator and streaming backends skip that set and log `Skipping FITS set`, so one tile that lacks the selected bands does not cost the tiles around it. Search the log for it before reading a short run as a complete one.
    - If the selection matches **no** set, every backend fails: that is a misspelled or inapplicable `selected_extensions`, not a gap in the data, and the error names both the bands you asked for and the bands that are there.
