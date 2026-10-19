# Building the documentation

From the repository root, use Python 3.13 (the version also used by GitHub
Actions and Read the Docs):

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

## GitHub Actions

`.github/workflows/build-docs.yml` builds HTML on pushes to every branch,
on pull requests, and on manual runs. It runs on all file changes because the
API reference also depends on Python source outside `docs/`.

Open the **Build documentation** run under the repository's **Actions** tab.
Download the `documentation-html` artifact, extract it, and open `index.html`.
Artifacts are retained for 14 days. Sphinx warnings and errors fail the workflow
and the Read the Docs build, so incomplete API pages cannot silently pass.

## Read the Docs hosting

Read the Docs uses `.readthedocs.yaml` and the same `docs/requirements.txt`.
The GitHub workflow provides a build check and artifact; Read the Docs publishes
the documentation through its GitHub integration.

After creating an account at <https://app.readthedocs.org/>:

1. Ask an existing maintainer for access to the `languagechange` project linked
   from the repository README. If the project has not been imported yet, add
   `ChangeIsKey/languagechange` with the GitHub integration. Connecting the
   repository may require an organization administrator.
2. Confirm that the project uses `.readthedocs.yaml` and the default branch is
   `main`. Check that the GitHub integration is connected.
3. For automatic hosting of new branches, add an automation rule with version
   type **Branch**, match **Any version**, and action **Activate version**.
   Activate existing branches in **Versions** as needed; the rule does not
   retroactively activate them.
4. Enable pull request builds in the project's settings if hosted previews
   are wanted.
5. Push a documentation change to a branch and verify both its GitHub Actions
   run and its Read the Docs build and version URL.

See the official [Git integration guide](https://docs.readthedocs.com/platform/stable/reference/git-integration.html)
and [automation rules guide](https://docs.readthedocs.com/platform/stable/automation-rules.html).
