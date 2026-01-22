# LanguageChange

LanguageChange is a Python toolkit for exploring lexical semantic change across corpora and time. It bundles data loaders, embedding pipelines, alignment strategies, and evaluation utilities so you can go from raw corpora to change scores and visual analyses in a single workflow.

## Key Features
- Ready-to-use benchmarks (SemEval 2020 Task 1, DWUG) plus helpers for your own corpora.
- Static and contextualised representation pipelines (count, PPMI, SVD, transformer-based) with caching.
- Alignment and comparison utilities (e.g. Orthogonal Procrustes) and standard change metrics such as PRT and APD.
- Plotting helpers for DWUG graphs and embeddings to inspect model behaviour visually.

## Installation

```bash
pip install languagechange
```

LanguageChange targets Python 3.8+ and depends on PyTorch, transformers, and several NLP/visualisation libraries. Installing inside a virtual environment is recommended.

## Quickstart

```python
from pathlib import Path
from languagechange.benchmark import SemEval2020Task1
from languagechange.models.representation.static import CountModel, PPMI, SVD

# Download the English SemEval 2020 Task 1 benchmark to the local cache
benchmark = SemEval2020Task1("EN")
corpus = benchmark.corpus1_lemma

artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)

# Build a count-based space, transform it with PPMI, then reduce with SVD
count = CountModel(corpus, window_size=10, savepath=artifacts / "corpus1_count")
count.encode()

ppmi = PPMI(count, shifting_parameter=5, smoothing_parameter=0.75, savepath=artifacts / "corpus1_ppmi")
ppmi.encode()

svd = SVD(ppmi, dimensionality=100, gamma=1.0, savepath=artifacts / "corpus1_svd")
svd.encode()
svd.load()

print(svd["plane_nn"])  # vector for a target lemma
```

More end-to-end walkthroughs live in the [examples](examples):
- [Compare static representations](examples/compare-representations/) across time slices.
- [Visualise DWUG usage graphs](examples/visualizations/) to inspect annotator judgements.

## Development Setup

Clone the repository and install an editable build with the project extras you need:

```bash
git clone https://github.com/ChangeIsKey/languagechange.git
cd languagechange
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Running the examples may require additional packages listed under each example directory.

## Documentation

For more detailed information, read the [API reference guide](https://languagechange.readthedocs.io/en/latest/).

## Citation

The library is under active development. If it supports your research, please cite it as:

```
@misc{languagechange,
  title = {LanguageChange: A Python library for studying semantic change},
  author = {{Change is Key!}},
  year = {n.d.}
}
```

## Credits

LanguageChange is developed by the [*Change is Key!*](https://www.changeiskey.org/) team with support from Riksbankens Jubileumsfond (grant M21-0021). Contributions and feedback are very welcome—feel free to open issues or pull requests.
