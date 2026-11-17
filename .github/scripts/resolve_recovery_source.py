"""Validate the source commit recorded by a successful release build."""

import json
import re
import subprocess
import sys
from pathlib import Path


def resolve(run_sha, manifest_path):
    source_sha = json.loads(Path(manifest_path).read_text())["commit"]
    if not all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{40}", x) for x in (run_sha, source_sha)):
        raise ValueError("Invalid source commit")
    if source_sha != run_sha:
        parents = subprocess.check_output(["git", "rev-list", "--parents", "-n", "1", source_sha], text=True).split()
        if parents != [source_sha, run_sha]:
            raise ValueError("Release source is not the run commit or its direct child")
        files = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", source_sha], text=True).splitlines()
        if files != ["pyproject.toml"]:
            raise ValueError("Automatic version commit changed unexpected files")
    return source_sha


if __name__ == "__main__":
    try:
        source = resolve(sys.argv[1], sys.argv[2])
        with Path(__import__("os").environ["GITHUB_OUTPUT"]).open("a") as output:
            print(f"source_sha={source}", file=output)
        print(f"Verified release source {source}")
    except (IndexError, KeyError, OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Recovery source check failed: {error}")
