# Release recovery preparation

The workflow does not publish to PyPI. It can create a GitHub release only after
the exact built files are confirmed on PyPI. A successful preview alone does
not mean a release was published.

The build job saves the wheel, source distribution, and
`release-evidence/manifest.json` together as an Actions artifact. The manifest
records the source commit, project name, version, filenames, and SHA-256 hashes.
The receiving job verifies the downloaded files against the checked-out source
and manifest, then compares them with PyPI. A PyPI version that exists with
different, missing, extra, or yanked files stops the preview for investigation.

For a future retry, preserve and reuse the exact artifact from the original
release run. Do not assume that a new build from the same commit has identical
bytes. GitHub Actions artifacts expire according to repository settings, so
long-term retention and recovery after expiry still need an agreed solution.

The `Inspect release recovery` workflow accepts the original run ID. It checks
that the run was a push to main, verifies its build job, downloads its original
artifact, and compares the files with the recorded commit and PyPI. By default
it only inspects. The optional create-GitHub-release input can create or verify
the matching GitHub release, but only when all PyPI files match. It requires
the original artifact to remain available and is not yet tested on GitHub
Actions. It never uploads to PyPI.

The release step checks an existing tag points to the original commit before
creating or accepting a release. Partial or conflicting PyPI uploads stop for
investigation. Before enabling PyPI uploads, test recovery on GitHub and agree
how to recover after artifact expiry or a partial PyPI upload.
