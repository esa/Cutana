---
name: New Release
about: Creating a new release version for Cutana. Only for Maintainers.
title: "Release "
labels: "release"
assignees: "gomezzz"
---

# Feature

## What Needs to Be Done (chronologically)

- [ ] Create PR from `main` -> `develop` to incorporate hotfixes / documentation changes that were made in `main` since the last release.
- [ ] Review the PR (if OK - merge, but DO NOT delete the branch, if problems arise they shall be fixed in `develop` not in `main`)
- [ ] Create a new branch from `develop` called `release` (e.g. `release-0.1.0`)
- [ ] Write changelog into `CHANGELOG.md`
- [ ] Minimize and update dependencies in `pyproject.toml` (regenerate `uv.lock` with `uv lock` if versions change). Keep `environment.yml` in sync — it is a compatibility shim for Euclid Datalab until they adopt uv.
- [ ] Check unit tests -> Check all tests pass on CPU (e.g. in Datalabs) and that there are tests for all important features
- [ ] Check documentation -> Check presence of documentation for all new or changed user-facing features in README.md
- [ ] Change version number in `pyproject.toml` and `__init__.py`
- [ ] Create PR: `release` → `main`, `release` -> `develop`
- [ ] Double-check that license header workflow passes on the PR
- [ ] Test that you can locally install the module (`uv sync --all-extras` or `pip install -e .`)
- [ ] Trigger the "Upload Python Package to testpypi" GitHub Action on the release branch
- [ ] Test the TestPyPI build: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cutana`
- [ ] Request and run PR Review
- [ ] Merge `release` into `main`, and `develop`
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR) reviewed
- [ ] Download the release from GitHub
- [ ] Prepare for release on the [public GitHub repo](https://github.com/ESA/Cutana) by removing non-essential folders and files in the downloaded version i.e currently `utility_scripts`, `.claude/commands`, `docs`, `CLAUDE.md`, `examples`, `scripts`
- [ ] Create a PR to the public repo with that version
- [ ] Merge the PR to the public repo
- [ ] Create a new release on the public repo
- [ ] Trigger the "Upload Python Package to PyPI" GitHub Action on the `main` branch
- [ ] Verify the PyPI release: `pip install cutana` (in a fresh environment)
