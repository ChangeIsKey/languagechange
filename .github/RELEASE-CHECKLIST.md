# Release checklist

## What has been tested

- The `testpypi-v0.3.0.dev1` tag built, uploaded, and installed a TestPyPI
  package without dependencies.
- On `ci/test-version-detection`, a change from `0.3.0` to `0.3.1` triggered a
  TestPyPI development release. A push without a version change was skipped.
- The test branch and its `0.3.1` version are experimental: do not merge that
  branch into main as the official version bump.

## Before an official release

1. Confirm the copyright year in `LICENSE` and remove its placeholder.
2. Review the release workflow, its permissions, and the recovery procedure.
3. Implement and test the official PyPI upload only after the TestPyPI process
   has been reviewed and approved. The current workflow does not upload to PyPI.
4. Configure PyPI Trusted Publishing and the GitHub publishing environment.
5. Test recovery, including partially uploaded files, and decide how to retain
   the original artifacts beyond their Actions expiration date.
6. Check the intended version is unused and update `pyproject.toml` in a
   reviewed change. Felix suggested the next official minor version, `0.3.0`.
7. After the change reaches main, verify tests, the PyPI files, the GitHub tag
   and release, and installation. Confirm both releases identify the same source.

The installation check used `--no-deps`; it confirms that the wheel is
installable and its version is correct, not that all dependencies or library
features work in a clean environment.
