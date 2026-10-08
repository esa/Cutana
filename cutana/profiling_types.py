#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Shared, dependency-free dataclasses for profiling / per-worker introspection.

These types live in their own module (with no other ``cutana`` imports) so any
component -- orchestrator, workers, benchmark scripts -- can depend on them
without risking an import cycle (issue #312).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


class Stage:
    """Canonical names of the profiled pipeline stages.

    These are the ``step`` keys ``PerformanceProfiler`` records and the labels
    ``ContextProfiler`` is given. Defining them here -- in the dependency-free
    shared module -- means workers, the profiler and the benchmark scripts all
    agree on one set of names instead of each repeating string literals (#312).
    Add a stage here (and to :data:`COMPUTE_STAGES` if a source passes through it)
    whenever a new ``ContextProfiler`` block is introduced.
    """

    FITS_LOADING = "FitsLoading"
    CUTOUT_EXTRACTION = "CutoutExtraction"
    IMAGE_RESIZING = "ImageResizing"
    CHANNEL_MIXING = "ChannelMixing"
    NORMALISATION = "Normalisation"
    METADATA_POSTPROCESSING = "MetaDataPostprocessing"
    ZARR_SAVING = "ZarrSaving"
    FITS_SAVING = "FitsSaving"


# Per-source compute stages, in pipeline order. The output-writer stages
# (ZARR_SAVING / FITS_SAVING) are deliberately excluded: the in-memory streaming
# path the profiler drives never writes to disk, so they do not fire there.
COMPUTE_STAGES = (
    Stage.FITS_LOADING,
    Stage.CUTOUT_EXTRACTION,
    Stage.IMAGE_RESIZING,
    Stage.CHANNEL_MIXING,
    Stage.NORMALISATION,
    Stage.METADATA_POSTPROCESSING,
)


@dataclass
class WorkerInfo:
    """Per-worker detail surfaced by ``StreamingOrchestrator.get_worker_info`` (#354).

    Spawn-time fields are set when the worker is started; the remaining fields are
    filled in from the worker's ``complete`` message. Defining every field here
    means all consumers know up front what they can read and write.

    Attributes:
        process_id: Worker process identifier.
        batch_index: Index into the orchestrator's internal batch ranges.
        n_sources: Number of sources assigned to this worker.
        pool_slot: Shared-memory pool slot index assigned to the worker.
        start_time: Spawn time (epoch seconds).
        end_time: Completion time (epoch seconds); ``None`` while still running.
        sources_per_fits_set: Source count keyed by FITS-set signature (a string,
            since this crosses the worker→parent JSON boundary), e.g.
            ``{"VIS.fits, NIR-H.fits": 30}``. ``None`` until completion. Both the
            distinct-set count (``len``) and the per-set distribution
            (``.values()``) derive from this, and keying by the set lets callers
            spot several workers hitting the same FITS sets.
        performance: Per-stage ``PerformanceProfiler.get_statistics()`` output
            (None until completion, or if the worker raised before reporting).
    """

    process_id: str
    batch_index: int
    n_sources: int
    pool_slot: int
    start_time: float
    end_time: Optional[float] = None
    sources_per_fits_set: Optional[Dict[str, int]] = None
    performance: Optional[Dict[str, Any]] = None

    # Keys a worker may report in its ``complete`` message that map onto fields here.
    COMPLETION_FIELDS = ("sources_per_fits_set", "performance")

    def merge_completion(self, batch_info: Dict[str, Any]) -> None:
        """Fill completion-time fields from a worker's ``batch_info`` dict.

        Only keys actually present are copied, so a worker that raised before its
        performance summary was assembled still merges whatever it did report.

        Args:
            batch_info: The ``batch_info`` payload from the worker's complete message.
        """
        for key in self.COMPLETION_FIELDS:
            if key in batch_info:
                setattr(self, key, batch_info[key])
