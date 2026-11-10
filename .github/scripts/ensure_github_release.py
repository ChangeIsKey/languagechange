"""Create or verify a GitHub release for a previously verified PyPI package."""

import json
import os
from pathlib import Path
import re
import sys
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def api(method, path, payload=None, missing_ok=False):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request('https://api.github.com' + path, data=body, method=method,
                  headers={'Accept': 'application/vnd.github+json',
                           'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                           'X-GitHub-Api-Version': '2022-11-28',
                           'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    sha = os.environ.get('RELEASE_SOURCE_SHA') or os.environ['GITHUB_SHA']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Invalid repository name')
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('Invalid source commit')
    with Path('pyproject.toml').open('rb') as file:
        version = tomllib.load(file)['project']['version']
    if not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version):
        raise ValueError('Invalid release version')
    tag = 'v' + version
    base = '/repos/' + repo
    ref = api('GET', base + '/git/ref/tags/' + tag, missing_ok=True)
    if ref is None:
        ref = api('POST', base + '/git/refs', {'ref': 'refs/tags/' + tag, 'sha': sha})
    obj = ref['object']
    seen = set()
    while obj['type'] == 'tag':
        tag_sha = obj['sha']
        if not re.fullmatch(r'[0-9a-f]{40}', tag_sha) or tag_sha in seen or len(seen) >= 10:
            raise ValueError('Invalid or recursive annotated tag')
        seen.add(tag_sha)
        obj = api('GET', base + '/git/tags/' + tag_sha)['object']
    if obj.get('type') != 'commit' or obj.get('sha') != sha:
        raise ValueError(f'Tag {tag} does not point to the verified source commit')

    release = api('GET', base + '/releases/tags/' + tag, missing_ok=True)
    if release is None:
        release = api('POST', base + '/releases', {'tag_name': tag, 'name': tag,
                                                   'generate_release_notes': True})
        print(f'Created GitHub release {tag}')
    else:
        if release.get('tag_name') != tag or release.get('draft') or release.get('prerelease'):
            raise ValueError('Existing GitHub release is not the expected public release')
        print(f'GitHub release {tag} already exists')


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, TypeError, tomllib.TOMLDecodeError) as error:
        sys.exit(f'GitHub release check failed: {error}')
