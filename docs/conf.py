"""Sphinx configuration for the ``languagechange`` project."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Ensure the project root can be imported when building docs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "languagechange"
author = "Change is Key!"

try:
    release = version("languagechange")
except PackageNotFoundError:  # fallback when the package is not installed in the environment
    release = "0.1.0"

# Version may differ from release by omitting patch-level info for short references.
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_mock_imports = [
    "IPython",
    "WordTransformer",
    "LSCDetection",
    "dload",
    "datasets",
    "filelock",
    "gensim",
    "huggingface_hub",
    "jsonlines",
    "jsonschema",
    "langchain",
    "langchain_core",
    "lxml",
    "matplotlib",
    "matplotlib.cm",
    "matplotlib.colors",
    "matplotlib.pyplot",
    "mlrose_hiive",
    "networkx",
    "numpy",
    "pandas",
    "peft",
    "platformdirs",
    "pydantic",
    "pyarrow",
    "pyarrow.csv",
    "pyarrow.parquet",
    "scipy",
    "sentence_transformers",
    "six",
    "sklearn",
    "sortedcontainers",
    "spacy",
    "torch",
    "transformers",
    "trankit",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = 'piccolo_theme'

# Configure autodoc defaults.
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Provide links to the standard library documentation.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
