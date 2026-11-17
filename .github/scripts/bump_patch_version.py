"""Prepare a patch bump while preserving an intentional version change."""

import argparse
from pathlib import Path
import re
import sys
import tomllib

VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def read_version(path):
    with path.open("rb") as source:
        value = tomllib.load(source)["project"]["version"]
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError(f"{path}: expected MAJOR.MINOR.PATCH")
    return value


def prepare(previous_path, current_path):
    previous = read_version(previous_path)
    current = read_version(current_path)
    old = tuple(map(int, previous.split(".")))
    new = tuple(map(int, current.split(".")))
    if new < old:
        raise ValueError(f"Version cannot decrease from {previous} to {current}")
    if new > old:
        return current, False

    updated = f"{new[0]}.{new[1]}.{new[2] + 1}"
    text = current_path.read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    if not section:
        raise ValueError("Expected an explicit [project] section")
    body = section.group(1)
    pattern = re.compile(r'(?m)^(\s*version\s*=\s*["\'])' + re.escape(current) + r'(["\']\s*(?:#.*)?)$')
    replaced, count = pattern.subn(lambda match: match.group(1) + updated + match.group(2), body)
    if count != 1:
        raise ValueError("Could not safely update the project version")
    current_path.write_text(text[:section.start(1)] + replaced + text[section.end(1):], encoding="utf-8")
    return updated, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    version, bumped = prepare(args.previous, args.current)
    print(f"Version: {version}; automatically increased: {str(bumped).lower()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as error:
        sys.exit(f"Patch version preparation failed: {error}")
