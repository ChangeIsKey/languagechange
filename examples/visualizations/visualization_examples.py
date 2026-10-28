"""Plot contextual word embeddings by year and save aligned results.

Run from a folder containing corpora/. --reuse-run reads previously saved
embeddings after checking the corresponding usage texts, offsets and dates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jsonlines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from languagechange.corpora import ParquetCorpus
from languagechange.models.meaning.clustering import Clustering
from languagechange.search import SearchTerm
from languagechange.usages import UsageDictionary
from languagechange.visualization import Visualizer, generate_cache_key, text_formatting

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--corpus", type=Path, default=Path("corpora/kubhist2-carlscronaswekoblad-1780.parquet"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization_native"))
parser.add_argument("--reuse-run", type=Path)
args = parser.parse_args()
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

all_usages = ParquetCorpus(str(args.corpus)).search(SearchTerm(lemma="man"))["lemma=man"]
usages = all_usages.group_by_time()
years = sorted(usages)
ordered_usages = [usage for year in years for usage in usages[year]]
boundaries = np.cumsum([0] + [len(usages[year]) for year in years])

if args.reuse_run:
    with jsonlines.open(args.reuse_run / "usages/lemma=man_usages.jsonl") as reader:
        previous = list(reader)
    assert [r["text"] for r in previous] == [u.text() for u in ordered_usages]
    assert [r["offsets"] for r in previous] == [u.offsets for u in ordered_usages]
    assert [r["time"] for r in previous] == [str(u.time) for u in ordered_usages]
    vectors = np.load(args.reuse_run / "usage_embeddings.npy", allow_pickle=False)
    embeddings = {year: vectors[boundaries[i]:boundaries[i + 1]] for i, year in enumerate(years)}
else:
    from languagechange.models.representation.contextualized import XL_LEXEME
    model = XL_LEXEME(device="cpu")
    embeddings = {year: model.encode(usages[year], batch_size=2) for year in years}
    vectors = np.concatenate([embeddings[year] for year in years])
assert vectors.shape == (len(ordered_usages), 1024)
assert np.isfinite(vectors).all()

clustering = Clustering(AgglomerativeClustering(
    n_clusters=None, metric="cosine", distance_threshold=0.5, linkage="complete",
))
labels = clustering.get_cluster_results(vectors).labels
visualizer = Visualizer(usages, embeddings=embeddings, cluster_labels=labels,
                        target="man", cache_dir=str(output_dir / "cache"))
assert np.array_equal(visualizer.embeddings, vectors)

# Use one shared legend below the panels so it does not cover the dots.
visualizer.plot_usage_embeddings(ncols=2, random_state=42,
                                plot_cluster_labels=False, plot=False)
figure = plt.gcf()
legend_entries = {}
for axis in figure.axes:
    handles, names = axis.get_legend_handles_labels()
    legend_entries.update(zip(names, handles))
names = sorted(legend_entries, key=int)
figure.legend([legend_entries[name] for name in names], names,
              title="Clusters", loc="lower center", ncol=9)
figure.tight_layout(rect=(0, 0.10, 1, 1))
graph_path = output_dir / "usage_embeddings_by_year.png"
figure.savefig(graph_path, dpi=150)
plt.close(figure)

# Read the exact coordinates saved by Visualizer, without recomputing t-SNE.
cache_key = generate_cache_key({"emb": vectors, "random_state": 42, "perplexity": 30})
coordinates = np.load(Path(visualizer.cache_mgr.cache_dir) / f"{cache_key}.npy", allow_pickle=False)
assert coordinates.shape == (len(ordered_usages), 2)
assert np.isfinite(coordinates).all()

UsageDictionary({"lemma=man": visualizer.usages}).save(str(output_dir / "usages"))
np.save(output_dir / "usage_embeddings.npy", vectors)
np.save(output_dir / "reduced_embeddings.npy", coordinates)
cluster_records = [
    {"usage_index": i, "year": str(year), "cluster": int(labels[i])}
    for group, year in enumerate(years)
    for i in range(int(boundaries[group]), int(boundaries[group + 1]))
]
with jsonlines.open(output_dir / "cluster_labels.jsonl", mode="w") as writer:
    writer.write_all(cluster_records)
with jsonlines.open(output_dir / "cluster_labels.jsonl") as reader:
    assert list(reader) == cluster_records
with jsonlines.open(output_dir / "usages/lemma=man_usages.jsonl") as reader:
    saved = list(reader)
assert [r["text"] for r in saved] == [u.text() for u in visualizer.usages]
assert [r["offsets"] for r in saved] == [u.offsets for u in visualizer.usages]
assert [r["time"] for r in saved] == [str(u.time) for u in visualizer.usages]
assert np.array_equal(np.load(output_dir / "usage_embeddings.npy", allow_pickle=False), vectors)
assert np.array_equal(np.load(output_dir / "reduced_embeddings.npy", allow_pickle=False), coordinates)

metadata = {
    "model": "pierluigic/xl-lexeme", "device": "cpu", "batch_size": 2,
    "embedding_source": "saved aligned embeddings" if args.reuse_run else "XL_LEXEME.encode",
    "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
    "year_counts": {str(year): len(usages[year]) for year in years},
    "clustering": {"metric": "cosine", "linkage": "complete", "distance_threshold": 0.5},
    "tsne": {"random_state": 42, "perplexity": 30, "metric": "cosine", "init": "random", "learning_rate": "auto"},
    "usage_count": len(vectors), "cluster_count": len(np.unique(labels)),
    "row_order": "Ascending year; original order within each year; usage_index is zero-based.",
}
(output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("Total usages:", len(visualizer.usages))
print("Year groups:", [(str(year), len(usages[year])) for year in years])
print("Embedding shape:", vectors.shape)
print("Number of clusters:", len(np.unique(labels)))
print("Reduced shape:", coordinates.shape)
print("Exports match:", True)
print("Cluster 0 examples:")
print("\n\n".join(text_formatting(visualizer.usages[int(i)], visualizer.usages[int(i)].time)
                  for i in np.flatnonzero(labels == 0)[:3]))
print("Output folder:", output_dir.resolve())
