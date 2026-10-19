# Building the documentation

From the repository root, use Python 3.13 (the version also used by Read the
Docs):

```sh
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -b html --fresh-env --fail-on-warning --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the result. Edit the `.rst` source
files, rather than the generated HTML. The API reference imports the local
source with external dependencies mocked, so building the docs does not require
installing the library's machine-learning dependencies or a PyPI release.

## Automatic builds on Read the Docs

Read the Docs builds and publishes the documentation when GitHub sends a push
notification through the project's webhook. It uses `.readthedocs.yaml` and
`docs/requirements.txt`; a separate GitHub Actions documentation build is not
required. Sphinx warnings and errors fail the Read the Docs build.

Maintainers can inspect the existing [languagechange project](https://app.readthedocs.org/projects/languagechange/)
and its [build history](https://app.readthedocs.org/projects/languagechange/builds/).
Request access to that project if it is not available in your dashboard.

The project settings needed for builds on every branch are:

1. A working GitHub integration for `ChangeIsKey/languagechange`. Under
   **Settings > Integrations**, inspect incoming webhook activity. A GitHub
   repository administrator may need to configure or re-sync the webhook.
2. An enabled automation rule with version type **Branch**, match **Any
   version**, and action **Activate version**. This enables new branch versions;
   activate existing branches separately under **Versions** when needed.
3. Default branch `main` and the repository-root `.readthedocs.yaml` configuration.

To verify automation, push a documentation change and check that a Read the
Docs build starts without clicking **Build** or **Rebuild**. Confirm that the
build uses the pushed commit, succeeds, and publishes a working **View docs**
link. For troubleshooting, inspect webhook delivery activity as well as the
build log.

Pull request previews can be enabled separately in **Pull request builds**.

See the official [Git integration guide](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
and [automation rules guide](https://docs.readthedocs.com/platform/stable/automation-rules.html).
