#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Merge multiple Cobertura coverage XML files into a single combined report.

Used by CI to combine coverage from build and browser test jobs.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def merge_coverage(input_files: list[Path], output: Path) -> None:
    files = [p for p in input_files if p.exists()]

    if not files:
        print("No coverage files found")
        sys.exit(1)

    if len(files) == 1:
        output.write_text(files[0].read_text())
        print(f"Using single coverage file: {files[0]}")
        return

    build_tree = ET.parse(files[0])
    browser_tree = ET.parse(files[1])
    build_root = build_tree.getroot()
    browser_root = browser_tree.getroot()

    build_packages = build_root.find(".//packages")
    existing_classes = {}
    for pkg in build_packages.findall("package"):
        for cls in pkg.findall(".//class"):
            filename = cls.get("filename")
            if filename:
                existing_classes[filename] = cls

    for pkg in browser_root.findall(".//packages/package"):
        for cls in pkg.findall(".//class"):
            filename = cls.get("filename")
            if not filename:
                continue
            if filename in existing_classes:
                build_cls = existing_classes[filename]
                build_lines = {ln.get("number"): ln for ln in build_cls.findall(".//line")}
                for line in cls.findall(".//line"):
                    num = line.get("number")
                    hits = int(line.get("hits", "0"))
                    if num in build_lines:
                        existing_hits = int(build_lines[num].get("hits", "0"))
                        build_lines[num].set("hits", str(max(existing_hits, hits)))
                    else:
                        lines_elem = build_cls.find("lines")
                        if lines_elem is not None:
                            lines_elem.append(line)
                all_lines = build_cls.findall(".//line")
                total = len(all_lines)
                hit = sum(1 for ln in all_lines if int(ln.get("hits", "0")) > 0)
                build_cls.set("line-rate", f"{hit / total:.4f}" if total > 0 else "0")
            else:
                pkg_name = pkg.get("name")
                target_pkg = None
                for bp in build_packages.findall("package"):
                    if bp.get("name") == pkg_name:
                        target_pkg = bp
                        break
                if target_pkg is None:
                    target_pkg = ET.SubElement(build_packages, "package", pkg.attrib)
                    ET.SubElement(target_pkg, "classes")
                target_pkg.find("classes").append(cls)
                existing_classes[filename] = cls

    all_lines = build_root.findall(".//line")
    total = len(all_lines)
    hit = sum(1 for ln in all_lines if int(ln.get("hits", "0")) > 0)
    build_root.set("line-rate", f"{hit / total:.4f}" if total > 0 else "0")

    build_tree.write(str(output), xml_declaration=True)
    print(f"Merged {len(files)} coverage files: {hit}/{total} lines covered")


if __name__ == "__main__":
    merge_coverage(
        input_files=[
            Path("build-results/coverage.xml"),
            Path("browser-results/coverage-browser.xml"),
        ],
        output=Path("combined-coverage.xml"),
    )
