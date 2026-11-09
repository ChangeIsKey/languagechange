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
    "sphinx_design",
    "sphinx_copybutton",
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

html_theme = "pydata_sphinx_theme"
html_title = "LanguageChange"
html_theme_options = {
    "logo": {
        "image_light": "languagechange-logo-light.svg",
        "image_dark": "languagechange-logo-dark.svg",
    },
    "navigation_depth": 3,
    "show_toc_level": 2,
    "collapse_navigation": True,
    "show_nav_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/ChangeIsKey/languagechange", "icon": "fa-brands fa-github"},
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/languagechange-icon.svg"

# Configure autodoc defaults.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
