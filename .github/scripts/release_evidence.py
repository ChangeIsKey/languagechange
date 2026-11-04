"""Record and verify the exact release files and source commit."""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evidence():
    commit = os.environ.get('RELEASE_SOURCE_SHA') or os.environ['GITHUB_SHA']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Expected a full 40-character source commit hash')
    with Path('pyproject.toml').open('rb') as file:
        project = tomllib.load(file)['project']
    files = sorted([*Path('dist').glob('*.whl'), *Path('dist').glob('*.tar.gz')])
    if len(files) != 2 or sum(p.name.endswith('.whl') for p in files) != 1:
        raise ValueError('Expected exactly one wheel and one source distribution')
    return {'commit': commit, 'name': project['name'], 'version': project['version'],
            'files': {path.name: sha256(path) for path in files}}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('create', 'verify'):
        raise ValueError('Expected create or verify')
    path = Path('release-evidence/manifest.json')
    expected = evidence()
    if sys.argv[1] == 'create':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + '\n')
        print('Recorded source commit and distribution checksums')
    else:
        recorded = json.loads(path.read_text())
        if recorded != expected:
            raise ValueError('Release evidence does not match the source or distribution files')
        print('Source commit and distribution checksums match release evidence')


if __name__ == '__main__':
    try:
        main()
    except (KeyError, ValueError, OSError, tomllib.TOMLDecodeError) as error:
        sys.exit(f'Release evidence check failed: {error}')
