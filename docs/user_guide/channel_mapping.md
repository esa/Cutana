[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Channel mapping and catalogue sampling

Use each input channel's name as a `channel_weights` key. Its value contains one
weight per output channel. Dictionary insertion order does not affect the result.
For example, `{"NIR-H": [0, 3], "VIS": [4, 0]}` produces `4 * VIS` followed by
`3 * NIR-H`, whichever order the catalogue uses.

When calling `combine_channels` directly, pass the `channel_names` returned by
`create_cutouts_direct`. Cutana resolves exact names first, then complete tokens
in FITS basename/HDU labels. Filter tokens treat `-` and `_` as interchangeable
in both directions, so a `NIR-H` key matches a `NIR_H` filename and vice versa.
Missing, ambiguous, and duplicate mappings fail. A single `PRIMARY` weight is
also supported for the single-channel extraction API, whose labels omit that HDU
suffix. Multiple HDUs require distinct, unambiguous keys.

## Discovery in the UI

Keep the same filters in the same order in your catalogue's `fits_file_paths`.
The UI displays that order and restores matrix cells by input name and output
index. Discovery checks up to 100 sampled rows for matching filter order and HDU
layout, caching each sampled FITS header once. It reads headers without loading
image arrays. Existing coordinate and footprint validation runs on the sample.

| Catalogue | Discovery population | Source count |
| --- | --- | --- |
| CSV | Random rows from the first 10,000 rows | Estimated when the prefix is full |
| Parquet | Random rows from the first 2,500 rows of up to four random row groups | Exact metadata count |
| FITS binary table | Random rows through a memory map | Exact header count |

Preview selection uses the same bounded reader, returning at most 10,000 rows.
It does not load the full catalogue before sampling. Repeated discovery uses a
fixed seed so that validation reports are reproducible.

> [!NOTE]
> CSV and Parquet discovery are not uniform samples of the entire catalogue.
> Uniform CSV sampling needs a scan or an index. Sampling cannot certify all
> rows. Runtime weight resolution uses the channel labels already in memory,
> protecting later batches without an additional catalogue scan or FITS open.

These bounds apply to discovery and previews. The existing processing
`CatalogueBatchReader` still preloads the Parquet table, and `CatalogueIndex`
stores row IDs for the full catalogue. Those processing components require
separate changes before you can rely on bounded memory for a billion-row run.

> [!NOTE]
> Automatic UI filter discovery uses the Euclid filename recognizer. A file it
> cannot classify is labelled `UNKNOWN`. One `UNKNOWN` tile per row works
> end to end: it is a single channel, so its weight pairs with it positionally
> and its name is never consulted. **Two or more do not** — they collapse onto
> one label, and weights and WCS are both looked up by name, so discovery
> refuses the catalogue. Rename the files so each band is identifiable, or use
> the Python API, where you pass channel labels explicitly.
>
> The label has to be a constant. `channel_weights` is one dictionary for the
> whole run, so a channel's name must mean the same thing in every row; anything
> read off an unrecognised tile's filename identifies that *tile* and changes
> from row to row.
