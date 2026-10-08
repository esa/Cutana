#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the throughput comparison in the release test report.

The report is read once per release, by someone deciding whether a slowdown is
real. A delta pinned to the wrong configuration, or a stale one presented as
current, is worse than no delta at all -- so what these tests care about is that
a number only ever appears next to the run it was measured from.
"""

import json
import re

import pytest
from release_test import (
    CH_IN_OUT_CONFIGS,
    build_config,
    format_throughput_delta,
    generate_markdown_report,
    load_throughput_baseline,
    run_key,
    worker_count_mismatches,
    write_throughput_baseline,
)

from cutana import get_default_config

CONFIG_FIELDS = {
    "catalogue_size": "small",
    "gen_type": "disk",
    "ch_in_out": "1vis1",
    "normalisation": "none",
    "output_format": "zarr",
    "data_type": "uint8",
    "target_resolution": 64,
}


def make_record(throughput=100.0, status="PASS", max_workers=8, **overrides):
    """Build a matrix run record, defaulting every field the report reads."""
    record = {
        **CONFIG_FIELDS,
        "max_workers": max_workers,
        "n_total": 10000,
        "elapsed": 100.0,
        "throughput": throughput,
        "check_stats": None,
        "wcs_stats": None,
        "status": status,
        "max_diff": 0.0,
        "mean_diff": 0.0,
    }
    record.update(overrides)
    return record


def split_row(line):
    """Split a markdown table row on its real delimiters.

    Two column headings contain an escaped pipe (``Max \\|Δ\\|``), so a plain
    ``split("|")`` reports more columns than the table has and the alignment check
    it feeds would fail on a correct report.
    """
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


@pytest.mark.parametrize("field", sorted(CONFIG_FIELDS))
def test_every_configuration_field_changes_the_key(field):
    """A run must not be paired with one that differs in any configured dimension."""
    other = "changed" if isinstance(CONFIG_FIELDS[field], str) else 4096
    assert run_key(make_record()) != run_key(make_record(**{field: other}))


def test_the_key_does_not_depend_on_field_order():
    """Records are built in different places; the key has to come out the same."""
    forwards = make_record()
    backwards = {key: forwards[key] for key in reversed(list(forwards))}
    assert run_key(backwards) == run_key(forwards)


def test_only_runs_that_passed_become_a_baseline(tmp_path):
    """A failed run measures the rate of doing something other than the work."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline(
        [
            make_record(gen_type="disk"),
            make_record(gen_type="direct", throughput=None, status="FAIL"),
            make_record(gen_type="mem-streaming", throughput=200.0, status="FAIL"),
        ],
        path,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))["runs"]
    assert list(saved) == [run_key(make_record(gen_type="disk"))]


def test_a_baseline_survives_the_round_trip(tmp_path):
    """What is written has to be what a later release reads back."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(throughput=123.5)], path)

    baseline = load_throughput_baseline(path)

    assert baseline["runs"][run_key(make_record())]["throughput"] == 123.5
    assert baseline["cutana_version"]


def test_no_baseline_is_not_an_error(tmp_path):
    """The first release to carry one has nothing to compare against."""
    assert load_throughput_baseline(tmp_path / "absent.json") is None


def test_a_file_that_is_not_a_baseline_is_refused(tmp_path):
    """Silently reporting no deltas would read as 'unchanged', which is a lie."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"runs": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="cutana_version"):
        load_throughput_baseline(path)


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (100.0, 150.0, ("+50.0", "+50.0%")),
        (100.0, 80.0, ("-20.0", "-20.0%")),
        (100.0, 100.0, ("+0.0", "+0.0%")),
    ],
)
def test_the_delta_is_signed_against_the_baseline(tmp_path, previous, current, expected):
    """Positive has to mean faster than the previous release, not merely different."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(throughput=previous)], path)

    assert (
        format_throughput_delta(make_record(throughput=current), load_throughput_baseline(path))
        == expected
    )


def test_a_configuration_the_baseline_lacks_is_blank_rather_than_zero(tmp_path):
    """A new or previously-impossible run has no change to report, not no change."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(gen_type="disk")], path)

    delta = format_throughput_delta(
        make_record(gen_type="direct", throughput=500.0), load_throughput_baseline(path)
    )

    assert delta == ("-", "-")


def test_the_delta_columns_appear_only_when_there_is_a_baseline(tmp_path):
    """Empty columns in a release report invite being read as zero change."""
    baseline_path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(throughput=100.0)], baseline_path)
    records = [make_record(throughput=150.0)]

    generate_markdown_report(records, tmp_path / "with.md", load_throughput_baseline(baseline_path))
    generate_markdown_report(records, tmp_path / "without.md", None)

    with_baseline = (tmp_path / "with.md").read_text(encoding="utf-8")
    without_baseline = (tmp_path / "without.md").read_text(encoding="utf-8")
    assert "Δ c/s" in with_baseline and "Δ %" in with_baseline
    assert "Δ c/s" not in without_baseline and "Δ %" not in without_baseline


def test_the_delta_lands_in_its_own_columns(tmp_path):
    """A column count that drifts silently misaligns every cell after it."""
    baseline_path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(throughput=100.0)], baseline_path)

    generate_markdown_report(
        [make_record(throughput=150.0)],
        tmp_path / "report.md",
        load_throughput_baseline(baseline_path),
    )

    lines = (tmp_path / "report.md").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("| # |"))
    row = next(line for line in lines if line.startswith("| 1 |"))
    columns = split_row(header)
    cells = split_row(row)

    assert len(cells) == len(columns)
    assert cells[columns.index("Δ c/s")] == "+50.0"
    assert cells[columns.index("Δ %")] == "+50.0%"


def test_the_report_names_the_release_it_compared_against(tmp_path):
    """A delta against an unnamed baseline cannot be checked or reproduced."""
    baseline_path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record()], baseline_path)
    baseline = load_throughput_baseline(baseline_path)

    generate_markdown_report([make_record()], tmp_path / "report.md", baseline)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert baseline["cutana_version"] in report
    assert baseline["created"] in report


def test_the_baseline_records_the_worker_count(tmp_path):
    """Without it a later release cannot tell whether the comparison is like for like."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(max_workers=64)], path)

    saved = json.loads(path.read_text(encoding="utf-8"))["runs"]
    assert saved[run_key(make_record())]["max_workers"] == 64


def test_a_baseline_without_worker_counts_is_refused(tmp_path):
    """An older baseline would otherwise be compared as if the counts had matched."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "cutana_version": "0.3.2",
                "created": "2026-09-11 10:21:30 UTC",
                "machine": "Linux (x86_64)",
                "runs": {run_key(make_record()): {"throughput": 100.0, "elapsed": 100.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_workers"):
        load_throughput_baseline(path)


def test_a_differing_worker_count_is_reported(tmp_path):
    """The confound has to be named, not left for the reader to notice."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(max_workers=64)], path)

    mismatches = worker_count_mismatches(
        [make_record(max_workers=8)], load_throughput_baseline(path)
    )

    assert len(mismatches) == 1
    assert "64 workers then, 8 now" in mismatches[0]


def test_matching_worker_counts_are_not_reported(tmp_path):
    """A warning on every fair comparison would train the reader to ignore it."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(max_workers=8)], path)

    assert (
        worker_count_mismatches([make_record(max_workers=8)], load_throughput_baseline(path)) == []
    )


def test_the_report_warns_when_the_worker_counts_differ(tmp_path):
    """A delta measured across a changed default must not read as a code change."""
    path = tmp_path / "baseline.json"
    write_throughput_baseline([make_record(max_workers=64, throughput=200.0)], path)
    baseline = load_throughput_baseline(path)

    generate_markdown_report([make_record(max_workers=8)], tmp_path / "warned.md", baseline)
    generate_markdown_report([make_record(max_workers=64)], tmp_path / "quiet.md", baseline)

    assert "[!WARNING]" in (tmp_path / "warned.md").read_text(encoding="utf-8")
    assert "[!WARNING]" not in (tmp_path / "quiet.md").read_text(encoding="utf-8")


def build_test_config(tmp_path, max_workers):
    """Build a config the way a matrix run does, varying only the worker override."""
    return build_config(
        catalogue_path=tmp_path / "catalogue.parquet",
        output_dir=tmp_path / "out",
        output_format="zarr",
        target_resolution=64,
        normalisation="none",
        ch_cfg=CH_IN_OUT_CONFIGS["1vis1"],
        data_type="uint8",
        max_workers=max_workers,
    )


def test_pinning_the_worker_count_reaches_the_config(tmp_path):
    """A flag that is threaded but never applied would silently keep the default."""
    assert build_test_config(tmp_path, max_workers=3).max_workers == 3


def test_not_pinning_leaves_the_version_default_alone(tmp_path):
    """The default is what a release report should measure; pinning is the exception."""
    default = build_test_config(tmp_path, max_workers=None).max_workers

    assert default == get_default_config().max_workers


@pytest.mark.parametrize("field", ["created", "machine"])
def test_a_baseline_missing_a_field_the_report_reads_is_refused(tmp_path, field):
    """The report footnote reads these, so a gap must fail before the matrix runs.

    Accepting the file here and raising in generate_markdown_report would throw the
    report away after several hours of measurement.
    """
    baseline = {
        "cutana_version": "0.3.2",
        "created": "2026-09-11 10:21:30 UTC",
        "machine": "Linux (x86_64)",
        "runs": {},
    }
    del baseline[field]
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_throughput_baseline(path)


def _one_run_baseline(max_workers=8, throughput=100.0):
    """A baseline holding exactly the run make_record() describes."""
    return {
        "cutana_version": "0.3.2",
        "created": "2026-09-11 10:21:30 UTC",
        "machine": "Linux (x86_64)",
        "runs": {run_key(make_record()): {"throughput": throughput, "max_workers": max_workers}},
    }


def test_a_crashed_run_is_not_reported_as_a_worker_mismatch():
    """A run with no throughput was never compared, so it cannot be a confounded pair.

    Its delta cell is already `-`; warning about its worker count would put a
    [!WARNING] in the report about a comparison that never happened.
    """
    crashed = make_record(throughput=None, status="FAIL", max_workers=None)

    assert worker_count_mismatches([crashed], _one_run_baseline()) == []


def test_a_failed_verification_gets_no_delta():
    """A run that produced the wrong pixels did not do the work the baseline measured.

    write_throughput_baseline already excludes these on the baseline side; the current
    side has to match or the two sides mean different things.
    """
    failed = make_record(throughput=200.0, status="FAIL")

    assert format_throughput_delta(failed, _one_run_baseline()) == ("-", "-")


def test_a_worker_mismatch_on_a_passing_run_is_still_reported():
    """The guard above must not swallow the case the warning exists for."""
    passing = make_record(throughput=200.0, max_workers=8)

    assert worker_count_mismatches([passing], _one_run_baseline(max_workers=1)) == [
        "run 1 (disk | 1vis1): 1 workers then, 8 now"
    ]
