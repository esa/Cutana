[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Contributing to Cutana

Thank you for your interest in improving Cutana! Cutana is developed by the ESAC
Data Science team at the European Space Agency (ESA) and released under the ESA
Public License (ESA-PL).

## Read this first: community governance

The rules that apply to all our projects (code of conduct, contribution
workflow, Contributor License Agreement, and licensing) live in one place:

> **[ESAC Data Science Community Governance](https://www.cosmos.esa.int/web/data-science/contributing-to-our-software)**

**Please read it before opening a pull request.** In particular:

- **A signed Contributor License Agreement (CLA) is required before we can merge
  your contribution.** Signing it confirms that the work is yours to give and
  grants ESA the right to redistribute it under the project license. You keep
  the copyright to your contribution, and you only need to sign once. The form,
  the submission address, and the full explanation are in the governance
  document.
- For **small changes** (typos, small fixes, documentation), open a pull request
  directly. For **larger changes**, open an issue first so we can scope the work
  together.
- Please report security vulnerabilities privately, not in a public issue.

The rest of this document covers only the technical specifics of contributing to
Cutana.

## Issues and bug reports

Open issues via the GitHub issue tracker and use the provided templates. Clear
reproduction steps, your environment details (OS, Python version, Cutana
version), and the expected versus actual behaviour help us resolve things
quickly.

## Development environment

Cutana uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.
uv is a fast drop-in replacement for `pip` and `venv` that installs dependencies
from the project lockfile, so everyone develops against exactly the same
versions. Install it once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv
```

Then set up the project:

```bash
uv sync --all-extras   # creates .venv and installs all dependencies
```

Prefix commands with `uv run` to execute them inside that environment (for
example `uv run pytest tests/`), or activate `.venv` manually if you prefer.
`uvx <tool>` runs a tool in a throwaway environment without adding it to the
project.

If you would rather not use uv, `environment.yml` provides an equivalent conda
environment.

## Branching model

Cutana uses a two-branch model:

- **`develop`** is the integration branch. **Target `develop` for features,
  fixes, and refactors.**
- **`main`** is reserved for releases, hotfixes, and documentation.

Stacked pull requests are fine. Branch from the same branch you intend to target.

## Coding conventions

- **Commits and branches:** follow
  [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`) and
  use matching branch prefixes (`feat/...`, `fix/...`, `docs/...`).
- **Formatting and linting:** code is linted and formatted with
  [Ruff](https://docs.astral.sh/ruff/). Run both before pushing:

  ```bash
  uvx ruff check .
  uvx ruff format .
  ```

- **License headers:** every source file must start with the standard ESA license
  header. Copy it from any existing file of the same type. CI rejects files
  without it.
- **Style:** follow PEP 8, use Google-style docstrings, and prefer plain
  functions over classes unless state genuinely needs managing. Comments should
  explain why, not restate what the code does.
- **Configuration:** configuration is accessed through DotMap (`cfg.a.b`).
  Validation belongs in `validate_config.py` and defaults in
  `get_default_config.py`.

## Tests

Add or update tests for your change, and make sure the suite passes locally
before pushing:

```bash
uv run pytest --cov=cutana tests/                            # backend tests with coverage
uv run pytest tests/ui/ -v                                   # UI widget tests
uv run pytest -m "browser and not slow" tests/browser/ -v    # browser smoke tests
```

The full browser suite (`-m "browser"`) is slow and is not run in CI, but is
worth running locally if you touch the UI.

## Continuous integration

Only pull requests that pass all CI checks are merged. CI runs:

- License-header check
- Ruff lint and format check
- Dead-code (Vulture) check
- Test suite, including browser smoke tests

Please make sure these pass locally first. CI is a safety net, not a substitute
for running the checks yourself.

## Questions

Not sure about something? Open an issue. We are happy to help.
