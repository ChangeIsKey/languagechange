# LanguageChange

LanguageChange is a Python toolkit for exploring lexical semantic change across corpora and time. It comes with data loaders, embedding pipelines, cluster and comparison methods, evaluation utilities as well as plotting helpers so you can go from raw corpora to change scores and visual analyses in a single workflow.

## Key Features
- Ready-to-use benchmarks (SemEval 2020 Task 1, DWUG) plus helpers for your own corpora.
- Static and contextualised representation pipelines (count, PPMI, SVD, transformer-based) with caching.
- Alignment and comparison utilities (e.g. Orthogonal Procrustes) and standard change metrics such as PRT and APD.
- Plotting helpers for DWUG graphs, embeddings clusters to inspect model behaviour visually.

## Installation

```bash
pip install langchange
```

LanguageChange targets Python 3.8+ and depends on PyTorch, transformers, and several NLP/visualisation libraries. Installing inside a virtual environment is recommended.

## Quickstart

LanguageChange can be used to analyze data from from scratch. 


```python

from languagechange.corpora import HistoricalCorpus, LinebyLineCorpus
from languagechange.search import SearchTerm
from sklearn.cluster import AgglomerativeClustering
from languagechange.models.change.metrics import JSD
from languagechange.models.representation.contextualized import XL_LEXEME
from languagechange.pipeline import CDPipeline
from languagechange.visualization import Visualizer


# Read two novels downloaded from gutenberg.org.
# Pride and prejudice (pp): https://www.gutenberg.org/cache/epub/42671/pg42671.txt
# Little women (lw): https://www.gutenberg.org/cache/epub/37106/pg37106.txt

# Loads the data, tokenizes and lemmatizes.
pp_corpus = LinebyLineCorpus("pp.txt", 1813)
lw_corpus = LinebyLineCorpus("lw.txt", 1868)

# Loads into HistoricalCorpus classes to compare across time.
novels = HistoricalCorpus([pp_corpus, lw_corpus])

# Define target word of interest.
target = "agreeable"
search_term = SearchTerm(lemma=target)
term = str(search_term)

# Retrieves the target usages.
pp_usages = pp_corpus.search(search_term)
lw_usages = lw_corpus.search(search_term)
all_usages = pp_usages + lw_usages

# Initialize model, cluster method and vector comparison metrics (JSD).
model = XL_LEXEME()
clustering_algorithm = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,
    linkage="complete",
    metric="cosine",
)
metric = JSD()

# Initializes pipeline
pipeline = CDPipeline(
    all_usages,
    model,
    metric,
    clustering_algorithm,
)

# Run the pipeline
sampled, embeddings, cluster_labels, change_scores = (
    pipeline.run_pipeline()
)


# Initializes the visualizer
visualizer = Visualizer(
    usages=sampled[term],
    embeddings=embeddings[term],
    cluster_labels=cluster_labels[term],
    target=term,
)

# Visualizes the usage embedding clusters
visualizer.plot_usage_embeddings()

```


Or be used with several predefined datasets. For example, the test data from [SemEval2020 task 1](https://aclanthology.org/2020.semeval-1.1/) for unsupervised lexical semantic change detection. See the [resources file](languagechange/resources_hub.json) for all our supported datasets.

```python

# Initializes a DWUG dataset
dwug = DWUG(dataset='DWUG', language='EN', version='3.0.0')

# Initialize the XL-LEXEME model
model = XL_LEXEME()

# Define a target word (with POS) to analyze.
target = "plane_nn"

# Retrieve the target usages for that word.
target_usages = dwug.get_word_usages(target) 

# Annotate and cluster the target word using XL-LEXEME
dwug.annotate_and_cluster(target, model)

```

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

For walkthrough tutorials and the API reference guide, please visit our official [documentation](https://languagechange.readthedocs.io/en/latest/). There are also documented notebooks and scripts in our [examples](examples/) folder.

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
