#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Merge multiple JUnit XML test result files into a single combined report.

Used by CI to combine test results from build and browser test jobs.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def merge_junit(input_files: list[Path], output: Path) -> None:
    files = [p for p in input_files if p.exists()]

    if not files:
        output.write_text('<testsuites><testsuite name="empty"/></testsuites>')
        return

    if len(files) == 1:
        output.write_text(files[0].read_text())
        return

    base = ET.parse(files[0])
    extra = ET.parse(files[1])
    base_root = base.getroot()

    for suite in extra.getroot():
        base_root.append(suite)

    tests = sum(int(s.get("tests", "0")) for s in base_root)
    errors = sum(int(s.get("errors", "0")) for s in base_root)
    failures = sum(int(s.get("failures", "0")) for s in base_root)
    time_total = sum(float(s.get("time", "0")) for s in base_root)
    base_root.set("tests", str(tests))
    base_root.set("errors", str(errors))
    base_root.set("failures", str(failures))
    base_root.set("time", f"{time_total:.3f}")

    base.write(str(output), xml_declaration=True)
    print(f"Merged JUnit XML: {tests} tests, {time_total:.1f}s total")


if __name__ == "__main__":
    merge_junit(
        input_files=[
            Path("build-results/pytest.xml"),
            Path("browser-results/pytest-browser.xml"),
        ],
        output=Path("combined-pytest.xml"),
    )
