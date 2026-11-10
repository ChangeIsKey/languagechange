"""Validate that a recovery target is a successful build from main."""

import json
import os
from pathlib import Path
import re
import sys


def main():
    if len(sys.argv) != 3:
        raise ValueError('Expected run and jobs JSON files')
    run = json.loads(Path(sys.argv[1]).read_text())
    jobs = json.loads(Path(sys.argv[2]).read_text())
    sha = run.get('head_sha')
    if (run.get('event') != 'push' or run.get('head_branch') != 'main'
            or run.get('path') != '.github/workflows/check-version.yml'
            or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha)):
        raise ValueError('The selected run is not a release workflow push on main')
    build = [job for job in jobs.get('jobs', []) if job.get('name') == 'check-version']
    if len(build) != 1 or build[0].get('status') != 'completed' or build[0].get('conclusion') != 'success':
        raise ValueError('The selected run does not have one successful check-version job')
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        print(f'source_sha={sha}', file=output)
    print(f'Validated release source commit {sha}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, TypeError) as error:
        sys.exit(f'Recovery check failed: {error}')
