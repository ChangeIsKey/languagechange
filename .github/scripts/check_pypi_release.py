"""Compare local distributions with a PyPI version JSON response.

The caller must treat network/authentication errors as errors, never as absence.
Pass --not-found only for an actual HTTP 404 from the version endpoint.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


def report(decision):
    print(decision)
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            print(f"decision={decision}", file=output)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path)
    source.add_argument("--not-found", action="store_true")
    source.add_argument("--lookup", action="store_true")
    args = parser.parse_args()

    files = sorted([*Path("dist").glob("*.whl"), *Path("dist").glob("*.tar.gz")])
    if len(files) != 2 or sum(p.name.endswith('.whl') for p in files) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution in dist/")
    local = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    if args.not_found:
        report("publish")
        return

    if args.lookup:
        with Path("pyproject.toml").open("rb") as config:
            project = tomllib.load(config)["project"]
        name, version = project["name"], project["version"]
        url = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
        try:
            with urlopen(url, timeout=30) as response:
                data = json.load(response)
        except HTTPError as error:
            if error.code != 404:
                raise
            report("publish")
            return
        if data.get("info", {}).get("name") != name or data.get("info", {}).get("version") != version:
            raise ValueError("PyPI response does not match the project name and version")
    else:
        data = json.loads(args.metadata.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("urls"), list) or not data["urls"]:
        raise ValueError("PyPI response has no release files; inspect it before retrying")
    if data.get("info", {}).get("yanked"):
        raise ValueError("PyPI release is yanked; inspect it before retrying")
    remote = {}
    for item in data["urls"]:
        if item.get("yanked"):
            raise ValueError("A PyPI release file is yanked; inspect it before retrying")
        name = item["filename"]
        digest = item["digests"]["sha256"]
        if name in remote or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("PyPI response has duplicate or invalid file metadata")
        remote[name] = digest

    if remote != local:
        raise ValueError("PyPI files differ from local files; stop and investigate")
    report("already-published")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        sys.exit(f"PyPI release check failed: {error}")
