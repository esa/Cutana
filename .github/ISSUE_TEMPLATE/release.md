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
- [ ] Run the release testing suite -> Run `uv run tests/release/release_test.py --matrix --output-folder PATH` to verify pixel-wise correctness and, on the FITS runs, the cutout WCS against the parent tiles (use `git lfs pull` to download the ground-truths cutouts and catalogues if necessary)
- [ ] Re-run each backend against a larger catalogue with `--catalogue-size medium --ch-in-out 1vis1 --normalisation log --output-format zarr --data-type uint8 --target-resolution 64` (and `--catalogue-size big` where disk allows) — ground truth at those sizes exists for that configuration only, so any other choice is refused before the run starts. **The matrix runs entirely on `small` (10k sources)**, so on its own it exercises no memory growth, batching or shared-memory pressure at survey scale, and a bug that only appears at 600k rows passes it
- [ ] Post the generated `.md` report as a comment on this issue, so the results are on the record before anything is published
- [ ] Check documentation -> Check presence of documentation for all new or changed user-facing features in README.md and `docs/`.
- [ ] Change version number in `pyproject.toml` and `__init__.py`
- [ ] Create PR: `release` → `main`, `release` -> `develop`
- [ ] Double-check that license header workflow passes on the PR
- [ ] Test that you can locally install the module (`uv sync --all-extras` or `pip install -e .`)
- [ ] Trigger the "Upload Python Package to testpypi" GitHub Action on the release branch
- [ ] Test the TestPyPI build: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cutana`
- [ ] Request and run PR Review
- [ ] Merge `release` into `main`, and `develop`
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR) reviewed
- [ ] Copy the release testing suite report from this issue into the GitHub release notes
- [ ] Promote this release's baseline so the next release's report can measure against it: copy `<output-folder>/throughput_baseline.json` from the `--matrix` run (next to `release_test_report.md`) to `tests/release/baselines/previous_release.json`. Add a `note` field if the run pinned `--max-workers`
- [ ] Download the release from GitHub
- [ ] Prepare for release on the [public GitHub repo](https://github.com/ESA/Cutana): copy the downloaded version onto a branch off the public `main`, then
  - **Remove** `.claude/`, `CLAUDE.md`, `REVIEW.md`, `claude_utils/`, `examples/`, `utility_scripts/`, `.gitattributes`, and the Git LFS release data `tests/release/ground_truth/` and `tests/release/catalogues/` (~20 GB)
  - **From `scripts/`, keep only** `merge_coverage_xml.py`, `merge_junit_xml.py`, `validate_browser_testing.py` (used by CI) and `gen_ref_pages.py` (used by the docs build)
  - **Keep** `docs/`, `zensical.toml` and `.github/workflows/docs.yml`: the docs site deploys from the public repo only
  - **Keep the public-only changes**: `CONTRIBUTING.md` and the README "Contributions welcome" badge; project URLs in `pyproject.toml` and `paper_scripts/README.md` pointing at `https://github.com/ESA/Cutana`; `(#N)` PR references removed from the new `CHANGELOG.md` section, because they point at the private repo
  - Confirm the result: diff the file list against the private tag, so the only differences are the ones listed above
- [ ] Create a PR to the public repo with that version
- [ ] Merge the PR to the public repo
- [ ] Create a new release on the public repo
- [ ] Trigger the "Upload Python Package to PyPI" GitHub Action on the `main` branch
- [ ] Verify the PyPI release: `pip install cutana` (in a fresh environment)
- [ ] Check that the online documentation is accessible and has been correctly generated