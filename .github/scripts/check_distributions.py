"""Check that the release distributions match the project metadata."""

import email.parser
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile


def check_metadata(raw, expected_name, expected_version, artifact):
    metadata = email.parser.BytesParser().parsebytes(raw)
    for key, expected in (("Name", expected_name), ("Version", expected_version)):
        actual = metadata.get_all(key, [])
        if actual != [expected]:
            raise ValueError(f"{artifact}: expected {key} {expected!r}, got {actual!r}")


def main():
    with Path("pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    wheels = list(Path("dist").glob("*.whl"))
    sources = list(Path("dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution in dist/")

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"{wheels[0]}: expected exactly one METADATA file")
        check_metadata(wheel.read(names[0]), project["name"], project["version"], wheels[0])

    with tarfile.open(sources[0], "r:gz") as source:
        names = [m for m in source.getmembers() if m.isfile() and m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        if len(names) != 1:
            raise ValueError(f"{sources[0]}: expected exactly one root PKG-INFO file")
        check_metadata(source.extractfile(names[0]).read(), project["name"], project["version"], sources[0])

    print(f"Both distributions match {project['name']} {project['version']}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        sys.exit(f"Distribution check failed: {error}")
