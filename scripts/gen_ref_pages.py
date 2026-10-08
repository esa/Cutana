#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Auto-generate API reference pages by walking the package tree."""

import shutil
from pathlib import Path

# Clear docs/api directory first to ensure no stale files
api_dir = Path("docs/api")
if api_dir.exists():
    shutil.rmtree(api_dir)
api_dir.mkdir(parents=True, exist_ok=True)

PACKAGES = ["cutana", "cutana_ui"]

# The nav points at docs/api/ as a single entry, so it needs an index listing every module
index_lines = ["# API reference\n"]

for package in PACKAGES:
    package_path = Path(package)
    index_lines.append(f"\n## `{package}`\n\n")
    for path in sorted(package_path.rglob("*.py")):
        if (
            path.name == "__init__.py"
            or path.name.startswith("_")
            or any(part.startswith(".") for part in path.parts)
        ):
            continue

        module_path = path.with_suffix("")
        doc_path = path.with_suffix(".md")
        full_doc_path = api_dir / doc_path

        # Ensure parent directory exists for the documentation file
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)

        parts = tuple(module_path.parts)

        # Write the API reference page
        with open(full_doc_path, "w", encoding="utf-8") as fd:
            fd.write(f"::: {'.'.join(parts)}\n")

        index_lines.append(f"- [`{'.'.join(parts)}`]({doc_path.as_posix()})\n")

with open(api_dir / "index.md", "w", encoding="utf-8") as fd:
    fd.writelines(index_lines)
