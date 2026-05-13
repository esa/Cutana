#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Launch the Cutana UI in Voila for interactive testing.

Starts a Voila server with the Cutana UI demo notebook, then
prints the URL. Use with Playwright MCP or a browser to interact.

The server writes its URL to ``tmp/voila_url.txt`` so Playwright MCP
(or scripts) can discover the port automatically.

Usage::

    python scripts/validate_browser_testing.py          # prints URL, waits
    python scripts/validate_browser_testing.py --open   # also opens browser
    python scripts/validate_browser_testing.py --port 8899  # fixed port
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
URL_FILE = REPO_ROOT / "tmp" / "voila_url.txt"


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Open the URL in a browser")
    parser.add_argument("--port", type=int, default=0, help="Fixed port (0 = auto)")
    args = parser.parse_args()

    notebook = REPO_ROOT / "examples" / "cutana_ui_demo.ipynb"
    if not notebook.exists():
        print(f"Notebook not found: {notebook}")
        sys.exit(1)

    port = args.port or _find_free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voila",
            str(notebook),
            f"--port={port}",
            "--Voila.ip=127.0.0.1",
            "--no-browser",
            "--enable_nbextensions=True",
            "--VoilaConfiguration.theme=dark",
            "--show_tracebacks=True",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://127.0.0.1:{port}"
    print(f"Starting Voila on port {port}...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 400:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Voila server failed to start within 60s")

    # Write URL to a well-known file so Playwright MCP can discover it
    URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    URL_FILE.write_text(url)

    print(f"\nVoila ready: {url}")
    print(f"URL written to: {URL_FILE}")
    print("Press Ctrl+C to stop.\n")

    if args.open:
        webbrowser.open(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=5)
    finally:
        URL_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
