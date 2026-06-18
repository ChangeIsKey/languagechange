import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
LOCALES_PATH = os.path.join(ROOT_PATH, "locales")
DEFINITION_TEMPLATE_PATH = os.path.join(LOCALES_PATH, "definition_templates")
RESOURCES_HUB_PATH = os.path.join(LOCALES_PATH, "resources_hub.json")
SPACY_PACKAGES_PATH = os.path.join(LOCALES_PATH, "spacy_package_names.json")