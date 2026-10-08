#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Unit tests for StreamingOrchestrator delivery accounting and failure surfacing.

These paths decide whether a worker that loses cutouts is reported or silently
accepted, so they need coverage that CI actually runs — the end-to-end regression
in ``tests/cutana/e2e/test_streaming_slow_consumer.py`` is marked ``slow`` and CI
runs ``-m "not browser and not slow"``.

Workers are stubbed rather than spawned: every path under test is driven purely by
a subprocess's exit code, its stdout messages and whether its stdin still accepts
writes, none of which needs a real FITS pipeline.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from cutana import get_default_config
from cutana.profiling_types import WorkerInfo
from cutana.streaming_orchestrator import StreamingOrchestrator


class _StubPipe:
    """Minimal stdin stand-in that can be made to fail like a closed pipe."""

    def __init__(self, broken: bool = False):
        self.broken = broken
        self.written = []

    def write(self, data):
        if self.broken:
            raise BrokenPipeError(32, "Broken pipe")
        self.written.append(data)

    def flush(self):
        if self.broken:
            raise BrokenPipeError(32, "Broken pipe")


class _StubStdout:
    """Stdout stand-in holding the lines a worker left in its pipe."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        while self._lines:
            yield self._lines.pop(0)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _StubProcess:
    """Stands in for subprocess.Popen for the orchestrator's polling paths."""

    def __init__(self, exit_code=None, stdin_broken=False, stdout_lines=None):
        self._exit_code = exit_code
        self.stdin = _StubPipe(broken=stdin_broken)
        self.stdout = None if stdout_lines is None else _StubStdout(stdout_lines)

    def poll(self):
        return self._exit_code

    def wait(self, timeout=None):
        return self._exit_code


@pytest.fixture
def orchestrator(tmp_path):
    """A real StreamingOrchestrator, stopping short of spawning workers.

    Constructed for real rather than with a stubbed ``__init__`` so these tests
    keep tracking the production attribute set instead of a hand-written copy of
    it that silently drifts. Only the few pieces ``init_streaming`` would set are
    filled in; none of these paths needs a spawned worker.
    """
    config = get_default_config()
    config.output_dir = str(tmp_path)
    catalogue = tmp_path / "catalogue.csv"
    catalogue.touch()
    config.source_catalogue = str(catalogue)

    orch = StreamingOrchestrator(config)
    orch._streaming_write_to_disk = False
    orch._max_workers = 1
    orch._batch_ranges = [[0, 1]]
    return orch


def _register_worker(orch, proc_id, assigned, chunks, reported=None, proc=None):
    """Wire up the per-worker bookkeeping a completed worker would have produced."""
    orch.active_processes[proc_id] = proc if proc is not None else _StubProcess(exit_code=0)
    orch._worker_to_pool[proc_id] = 0
    orch._pool_to_worker[0] = proc_id
    orch._worker_cutouts[proc_id] = chunks
    orch._worker_metadata[proc_id] = [{"source_id": str(i)} for i in range(sum(map(len, chunks)))]
    orch._worker_source_counts[proc_id] = assigned
    # The completion path stamps end_time on the record _spawn_worker creates (#354).
    # These tests drive _complete_worker directly, so the spawn-time record is seeded here.
    orch._worker_info[proc_id] = WorkerInfo(
        process_id=proc_id,
        batch_index=0,
        n_sources=assigned,
        pool_slot=0,
        start_time=time.time(),
    )
    if reported is not None:
        orch._worker_reported_totals[proc_id] = reported


def test_delivery_matching_assignment_records_no_shortfall(orchestrator):
    """A worker that hands back one cutout per source is not flagged."""
    chunk = np.zeros((4, 2, 2, 1), dtype=np.float32)
    _register_worker(orchestrator, "w0", assigned=4, chunks=[chunk], reported=4)

    orchestrator._complete_worker("w0")

    assert orchestrator._worker_shortfalls == []
    assert len(orchestrator._ready_results) == 1


def test_short_delivery_is_recorded_and_named_in_the_summary(orchestrator):
    """A worker delivering fewer cutouts than sources must be attributable.

    Regression for the reported failure: the run previously died with "no cutouts
    remain for the expected batch", which named neither the worker nor the count.
    """
    chunk = np.zeros((3, 2, 2, 1), dtype=np.float32)
    _register_worker(orchestrator, "w0", assigned=10, chunks=[chunk], reported=3)

    orchestrator._complete_worker("w0")

    assert orchestrator._worker_shortfalls == [("w0", 10, 3)]
    summary = orchestrator._shortfall_summary()
    assert "w0" in summary
    assert "3/10" in summary
    assert "7" in summary


def test_shortfall_summary_truncates_but_reports_the_remainder(orchestrator):
    """Long shortfall lists stay readable without hiding how many were omitted."""
    orchestrator._worker_shortfalls = [(f"w{i}", 10, 9) for i in range(15)]

    summary = orchestrator._shortfall_summary()

    assert "w0: 9/10" in summary
    assert "5 more" in summary
    assert "15 worker(s)" in summary


def test_nonzero_worker_exit_raises_and_points_at_both_logs(orchestrator, tmp_path):
    """A crashed worker must fail the run, naming the logs holding its traceback.

    The worker writes its traceback through loguru (shared session log) and to
    raw stderr (``<id>_stderr.log``), so both are named — pointing at only one
    risks sending users to the file they do not have.
    """
    _register_worker(orchestrator, "w0", assigned=4, chunks=[], proc=_StubProcess(exit_code=1))
    orchestrator._streaming_write_to_disk = True

    with pytest.raises(RuntimeError, match="failed with exit code 1") as excinfo:
        orchestrator._poll_workers()

    assert f"cutana_{orchestrator.config.session_timestamp}.log" in str(excinfo.value)
    assert "w0_stderr.log" in str(excinfo.value)
    assert "w0" not in orchestrator.active_processes


def test_worker_error_payload_is_quoted_in_the_failure(orchestrator):
    """The worker's own error must reach the user, not just a log path.

    A failing worker prints its exception to the protocol pipe before exiting;
    that message was parsed and dropped, so the run reported an exit code and
    told the user to go read a file.
    """
    _register_worker(orchestrator, "w0", assigned=4, chunks=[], proc=_StubProcess(exit_code=1))
    orchestrator._handle_worker_line(
        "w0", '{"processed_count": 0, "total_count": 4, "error": "FITS file not found: a.fits"}'
    )

    message = orchestrator._worker_failure_message("w0", 1, "its cutouts were not delivered")

    assert "FITS file not found: a.fits" in message


def test_worker_error_still_in_the_pipe_is_recovered(orchestrator):
    """The error line is often unread when the exit code is noticed.

    The worker prints it immediately before exiting, so the failure path has to
    drain the pipe rather than assume the poll loop already saw it.
    """
    proc = _StubProcess(
        exit_code=1,
        stdout_lines=['{"processed_count": 0, "total_count": 4, "error": "shm pool full"}\n'],
    )
    _register_worker(orchestrator, "w0", assigned=4, chunks=[], proc=proc)

    assert orchestrator._collect_worker_error("w0") == "shm pool full"


def test_unrecognized_worker_message_raises(orchestrator):
    """An unknown protocol message must fail hard rather than be dropped."""
    _register_worker(orchestrator, "w0", assigned=4, chunks=[])

    with pytest.raises(RuntimeError, match="Unrecognized message from worker w0"):
        orchestrator._handle_worker_line("w0", '{"type": "something_new"}')


def test_transport_mismatch_does_not_queue_a_result(orchestrator):
    """A worker whose handover is known incomplete must not queue a result.

    Accounting runs before the append so the caller never receives a batch
    assembled from a transfer the orchestrator has already judged broken.
    """
    chunk = np.zeros((2, 2, 2, 1), dtype=np.float32)
    _register_worker(orchestrator, "w0", assigned=5, chunks=[chunk], reported=5)

    with pytest.raises(RuntimeError, match="reported 5 cutouts but only 2"):
        orchestrator._complete_worker("w0")

    assert orchestrator._ready_results == []


def _write_worker_stdout_log(orchestrator, proc_id, payload):
    """Write the JSON summary line a disk-mode worker leaves in its stdout log."""
    log_dir = Path(orchestrator.config.output_dir) / "logs" / "subprocesses"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{proc_id}_stdout.log").write_text(json.dumps(payload) + "\n")


def test_disk_worker_shortfall_is_accounted_from_its_stdout_log(orchestrator):
    """Disk mode must account deliveries too, not report a permanent all-clear.

    Disk workers have no stdout pipe, so their counts were never recorded and
    every disk-mode failure claimed no worker had reported a shortfall.
    """
    orchestrator._streaming_write_to_disk = True
    _register_worker(orchestrator, "w0", assigned=10, chunks=[], proc=_StubProcess(exit_code=0))
    _write_worker_stdout_log(orchestrator, "w0", {"processed_count": 7, "total_count": 10})

    orchestrator._complete_worker("w0")

    assert orchestrator._worker_shortfalls == [("w0", 10, 7)]
    assert "w0: 7/10" in orchestrator._shortfall_summary()


def test_failed_worker_queues_no_result(orchestrator):
    """A dead worker's partial output must not reach the caller.

    Its zarr batch may never have been written and its cutout list is short by an
    unknown amount, so a caller that catches the failure and calls next_batch()
    again would otherwise be handed the wreckage.
    """
    orchestrator._streaming_write_to_disk = True
    _register_worker(orchestrator, "w0", assigned=10, chunks=[], proc=_StubProcess(exit_code=1))

    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        orchestrator._poll_workers()

    assert orchestrator._ready_results == []
    assert "w0" not in orchestrator._worker_source_counts


def test_disk_worker_without_a_summary_raises(orchestrator):
    """A disk worker cannot exit 0 without printing its summary.

    Absorbing the missing summary would drop the whole batch out of the
    accounting unnoticed — the silent shortfall this accounting exists to catch.
    """
    orchestrator._streaming_write_to_disk = True
    _register_worker(orchestrator, "w0", assigned=10, chunks=[], proc=_StubProcess(exit_code=0))

    with pytest.raises(RuntimeError, match="exited successfully but wrote no summary"):
        orchestrator._complete_worker("w0")


def test_disk_worker_error_is_read_back_from_its_stdout_log(orchestrator):
    """A disk worker's failure reaches the user through its redirected stdout."""
    orchestrator._streaming_write_to_disk = True
    _register_worker(orchestrator, "w0", assigned=10, chunks=[], proc=_StubProcess(exit_code=1))
    _write_worker_stdout_log(
        orchestrator, "w0", {"processed_count": 0, "total_count": 10, "error": "out of memory"}
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        orchestrator._poll_workers()


def test_delivery_report_exposes_the_totals_it_warns_about(orchestrator):
    """The shortfall warning tells callers to judge for themselves; give them the data.

    A deficit small enough to fit inside the final batch never raises, so
    without an accessor the warning pointed at totals nobody could read.
    """
    chunk = np.zeros((8, 2, 2, 1), dtype=np.float32)
    _register_worker(orchestrator, "w0", assigned=10, chunks=[chunk], reported=8)

    orchestrator._complete_worker("w0")

    assert orchestrator.get_delivery_report() == {"missing": 2, "shortfalls": [("w0", 10, 8)]}


def test_ack_on_dead_worker_reports_lost_cutouts(orchestrator, monkeypatch):
    """A broken ACK pipe must surface as lost cutouts, not a bare BrokenPipeError."""

    class _Pool:
        def read_slots(self, n):
            return np.zeros((n, 2, 2, 1), dtype=np.float32)

    _register_worker(
        orchestrator, "w0", assigned=4, chunks=[], proc=_StubProcess(stdin_broken=True)
    )
    orchestrator._shm_pools = [_Pool()]

    with pytest.raises(RuntimeError, match="exited before its chunk could be acknowledged"):
        orchestrator._handle_chunk_ready("w0", {"slot_count": 2, "metadata": [{}, {}]})


def test_complete_message_without_total_is_not_defaulted(orchestrator):
    """A malformed 'complete' must not silently become reported=0.

    A zero default would trip the transport-mismatch check and invent a failure.
    """
    _register_worker(orchestrator, "w0", assigned=4, chunks=[])

    with pytest.raises(KeyError, match="total_cutouts"):
        orchestrator._handle_worker_line("w0", '{"type": "complete"}')


def test_complete_line_left_in_the_pipe_completes_the_worker_once(orchestrator, tmp_path):
    """A 'complete' still buffered when the worker exits is applied exactly once.

    ``_try_read_worker`` takes a single line per poll, so a worker can exit with
    its 'complete' unread. ``_poll_workers`` therefore drains the pipe before
    finishing an exited worker — but ``_handle_worker_line`` *itself* completes
    the worker on 'complete', so finishing it again afterwards popped
    ``_worker_source_counts`` twice and raised ``KeyError``, killing
    mem-streaming runs mid-batch.

    A real file is used as stdout because the exit path goes through
    ``select()``, which needs a genuine descriptor.
    """
    chunk = np.zeros((3, 2, 2, 1), dtype=np.float32)
    stdout_path = tmp_path / "w0_stdout"
    stdout_path.write_text(
        # First line is consumed by _try_read_worker, leaving 'complete' buffered
        # for the drain — the ordering that triggered the double completion.
        json.dumps({"processed_count": 3, "total_count": 3})
        + "\n"
        + json.dumps({"type": "complete", "total_cutouts": 3, "batch_info": {}})
        + "\n"
    )
    with stdout_path.open() as stdout:
        proc = _StubProcess(exit_code=0)
        proc.stdout = stdout
        _register_worker(orchestrator, "w0", assigned=3, chunks=[chunk], proc=proc)

        orchestrator._poll_workers()

    assert orchestrator._worker_shortfalls == []
    assert len(orchestrator._ready_results) == 1
    assert orchestrator._worker_reported_totals.get("w0") is None, (
        "reported total should have been consumed by the single completion"
    )
    assert "w0" not in orchestrator._worker_source_counts
    assert "w0" not in orchestrator.active_processes
