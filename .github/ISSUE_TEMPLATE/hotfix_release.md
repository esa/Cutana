---
name: Hotfix Release
about: Creating a hotfix (patch) release for Cutana. Only for Maintainers.
title: "Hotfix Release "
labels: "release"
assignees: "gomezzz"
---

# Hotfix

A hotfix is a small, targeted patch (bug or security fix) shipped on top of the
latest release without the full release cycle. For feature releases use the
`New Release` template instead.

## What Needs to Be Done (chronologically)

- [ ] Ensure the fix itself is merged into `main` via a small PR straight to `main` (do NOT branch a full `release` branch for a hotfix)
- [ ] Bump the patch version in `pyproject.toml` and `__init__.py` (e.g. `0.3.0` -> `0.3.1`)
- [ ] Regenerate `uv.lock` with `uv lock` if any dependency changed, and keep `environment.yml` in sync (compatibility shim for Euclid Datalab until they adopt uv)
- [ ] Add a `## [vX.Y.Z] – YYYY-MM-DD` entry to `CHANGELOG.md` describing the fix
- [ ] Run the unit tests to confirm the fix and that nothing regressed
- [ ] Commit the version bump + changelog and push to `main`
- [ ] Trigger the "Upload Python Package to testpypi" GitHub Action on `main`
- [ ] Test the TestPyPI build: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cutana`
- [ ] Create the Release on GitHub from the version-bump commit, pasting the `CHANGELOG.md` entry as the release notes
- [ ] Trigger the "Upload Python Package to PyPI" GitHub Action on the `main` branch
- [ ] Verify the PyPI release: `pip install cutana` (in a fresh environment)
- [ ] Open a PR `main` -> `develop` to carry the fix + version bump back (DO NOT delete `main`; any follow-up fixes go through `develop`)
- [ ] Mirror to the public repo if the fix originated internally: apply the same removals and public-only changes as the "Prepare for release on the public GitHub repo" step in the release template (`.github/ISSUE_TEMPLATE/release.md`), open a PR to the [public GitHub repo](https://github.com/ESA/Cutana), merge it, and create the matching release there
