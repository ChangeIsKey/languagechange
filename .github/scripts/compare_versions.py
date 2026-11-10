"""Compare release versions and expose the result to GitHub Actions."""

import os
from pathlib import Path
import re
import sys
import tomllib


def read_version(path):
    with Path(path).open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    if not isinstance(version, str) or not re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version
    ):
        raise ValueError(f"{path}: expected a MAJOR.MINOR.PATCH version, got {version!r}")
    return version


def main():
    if len(sys.argv) != 3:
        raise ValueError("Expected the previous and current pyproject.toml paths")
    previous, current = map(read_version, sys.argv[1:])
    old_numbers = tuple(map(int, previous.split(".")))
    new_numbers = tuple(map(int, current.split(".")))
    if new_numbers < old_numbers:
        raise ValueError(f"Version cannot decrease from {previous} to {current}")

    changed = str(previous != current).lower()
    print(f"Previous version: {previous}\nCurrent version: {current}\nVersion changed: {changed}")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        print(f"changed={changed}\nversion={current}", file=output)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError, tomllib.TOMLDecodeError) as error:
        sys.exit(f"Version check failed: {error}")
