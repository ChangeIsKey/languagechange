# Use case checks in GitHub Actions

This is the setup for issue #25. `usecases.yml` runs the seven
use cases on each push and on an explicit workflow dispatch. There are
no path filters or branches that bypass these checks. The existing tests and
representation-comparison example workflows remain unchanged.

| Topic | Entry point | Inputs and execution |
| --- | --- | --- |
| Corpora | `examples/corpora/corpora_examples.py` | The documented Parquet search, inspection, sampling and native exports |
| Usages | `examples/usages/usages_examples.py` | Target offsets, full-date/year grouping, full-year intervals and sampling |
| Benchmarks | `examples/benchmarks/benchmark_examples.py` | WiC, XL-WSD, DWUG, NorDiaChange, DWUG Sense, conversions, evaluations and XL-LEXEME annotation/clustering |
| Definitions | `examples/definitions/definition_t5_xl_examples.py` | Fresh T5 XL generation followed by MPNet encoding; CPU in GitHub Actions, CUDA for ordinary generation |
| Prompting | `examples/prompting/prompting_groq_examples.py` | Four real Groq requests: two DURel scores and two WiC decisions |
| Time series | `examples/timeseries/timeseries_examples.py` | Fresh XL-LEXEME embeddings, APD and both JSD approaches; 10 usages in CI, 100 normally |
| Visualization | `examples/visualizations/visualization_examples.py` | Fresh embeddings, clustering, t-SNE, graph and context inspection; 10 usages in CI, 480 normally |

These are the use case entry points, including the Parquet route
for corpora. The optional XML conversion/raw-text processing route and the
older Desktop Llama/local-prompt variants are not substituted for these
entry points. The benchmark case loads the real resources in both modes. In CI it selects up
to ten records per split before WSD-to-WiC conversion and annotates ten egg
usages. Ordinary execution retains the full conversion and annotation.

## GitHub Actions CPU mode

GitHub sets `GITHUB_ACTIONS=true` automatically. The scripts check that value
with `os.environ.get("GITHUB_ACTIONS") == "true"`. The runner hides CUDA devices
from CI child processes, and every workflow case uses a standard Ubuntu CPU
runner. No GPU runner variable or self-hosted runner is needed.

Time series and visualization select ten contexts across all four years:
3 from 1780, 3 from 1781, 2 from 1782 and 2 from 1784, with seed 42. Keeping
all four years lets the APD/JSD and plotting pipelines work on the small run.
The t-SNE perplexity becomes 9 for ten rows. Outside GitHub Actions, these
scripts retain their normal data sizes and existing CPU settings.

Definitions still use the exact T5 XL model and ten plane contexts. CI
permits CPU generation; ordinary generation still requires CUDA and 13 GiB
of free GPU memory. Generation and MPNet encoding run in separate processes
so they do not keep both models loaded at the same time. There is no model
substitution or replay of saved predictions.

T5 XL generation and MPNet encoding passed on a standard Ubuntu 24.04 CPU
runner on 6 November 2026. The run generated ten definitions and checked a
finite 10 × 768 embedding array. The generation/encoding stages took about
three minutes including their model and dataset downloads. The complete job
also includes dependency installation and model-cache handling.
[Hosted run and downloadable results](https://github.com/ChangeIsKey/languagechange/actions/runs/42195029400).

Ten examples reduce inference work rather than model memory. The workflow
keeps a 120-minute timeout and fails if loading or execution cannot complete;
it does not silently substitute a smaller model.

Configure repository Actions secrets `GROQ_API_KEY` and `LANGSMITH_API_KEY`
privately in GitHub settings. The current Groq provider also requests
LangSmith tracing. Missing keys fail with the secret names only. Prompting
continues to make four real remote requests; Groq inference runs at the
provider, rather than on the CPU runner.

The lighter corpora/usages cases retain all 480 records because they do not
load a neural model. Their full-year interval and export checks stay intact.
There is no `continue-on-error` or saved-response fallback.

The use case workflow uses `push` rather than `pull_request` to avoid
duplicate model runs. Every branch pushed to this repository triggers it.
The existing PR test/example workflows remain enabled. Forks need their own
API secrets for the prompting case.

## Data and caching

`data_manifest.json` records the existing 1.9 MB Parquet input's SHA-256,
source, license and expected counts. The runner copies that committed file
into the new working directory and verifies its checksum. It needs no
Desktop paths, downloads folder, or local virtual environment.

The Parquet file is a conversion of Språkbanken (2024), *Kubhist 2:
Carlscronas Wekoblad 1780-talet*, DOI
[10.23695/4qx9-j028](https://doi.org/10.23695/4qx9-j028), under CC BY 4.0.
The conversion preserves the source text and annotations. It is already
tracked at `docs/tutorials/data/kubhist2-carlscronaswekoblad-1780.parquet`;
there is no second copy of it in the repository.

Benchmark resources download through the library's resource manager using
the versions selected in the example. Some resources are named
`no-version` upstream, so their remote contents may change. Dataset download
errors fail the corresponding run. Dataset licenses remain applicable.
The definitions run downloads DWUG EN 3.0.0 from its recorded resource.

Only public Hugging Face model weights are cached by the use case workflow.
Embeddings, generated definitions, Groq responses and t-SNE coordinates are
computed again. Each command requires a fresh output directory and the
examples use run-local result caches, for both local and GitHub Actions execution.
The large T5 XL weights are not uploaded as an artifact or stored in Git.

## Local commands

From the repository root, using a suitable environment:

```sh
python -m pip install . -r examples/_ci/requirements.txt
python examples/_ci/run_usecase.py corpora --output-dir outputs/check-corpora
python examples/_ci/run_usecase.py usages --output-dir outputs/check-usages
python examples/_ci/run_usecase.py benchmarks --output-dir outputs/check-benchmarks
python examples/_ci/run_usecase.py prompting --output-dir outputs/check-prompting
python examples/_ci/run_usecase.py timeseries --output-dir outputs/check-timeseries
python examples/_ci/run_usecase.py visualization --output-dir outputs/check-visualization
python examples/_ci/run_usecase.py definitions --output-dir outputs/check-definitions
```

Ordinary definitions generation requires the GPU environment. To reproduce
CI's small CPU mode locally, prefix a command with `GITHUB_ACTIONS=true`:

```sh
GITHUB_ACTIONS=true python examples/_ci/run_usecase.py timeseries --output-dir outputs/check-small-timeseries
```

Do not run T5 XL on a machine without sufficient memory. Run prompting only
after privately configuring both keys. Choose a new directory for each run;
do not use `--reuse-run` for CI checks.

## What the checks establish

The runner checks native JSONL parsing, record counts, text/offset/date
alignment, full-year intervals, finite arrays and scores, PNG validity, and
the required exported files. APD is recomputed from the new embeddings;
JSD is recomputed from the new cluster distributions. The two JSD approaches
must agree. Benchmark gold-label self-checks must succeed.

Groq responses are checked for valid schema/range, not exact prediction
values. Definition text is checked for nonempty strings and its embedding
shape, not agreement with a previously generated sentence. This is execution
coverage rather than an accuracy evaluation, and it does not guarantee
network services will always be available.

Each case writes `run.json` with its status, script/input hashes, checks and
failure details. The workflow uploads results and redacted logs for seven
days, even on a failed execution. Model caches are excluded. The complete
DWUG-context definition table stays local to the runner; its artifact
contains the prediction-only file and embeddings instead.

## Validate the CI support code

```sh
python -m pytest -q tests/test_usecases_ci.py
```

These focused tests exercise CI mode selection, invalid outputs, missing
prerequisites, failure propagation and data integrity. They are separate
from the hosted model executions.

All seven cases passed in the hosted run linked above on 6 November 2026.
The prompting job was retried after its Groq secret was added; its retry
made fresh API requests and passed the response/export checks. The ordinary
test and representation-comparison workflows also passed on this branch.
