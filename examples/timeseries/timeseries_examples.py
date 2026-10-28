"""Compute APD and JSD time series and save actual results.

Run from the directory containing corpora/. Use --reuse-run to recompute
scores from a saved sample and its aligned embeddings.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import jsonlines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from languagechange.corpora import ParquetCorpus
from languagechange.models.change.metrics import APD, JSD
from languagechange.models.change.timeseries import TimeSeries
from languagechange.models.meaning.clustering import Clustering
from languagechange.search import SearchTerm
from languagechange.usages import UsageDictionary
from languagechange.utils import _parse_year

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path('outputs/timeseries_native'))
parser.add_argument('--reuse-run', type=Path)
args = parser.parse_args()
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
corpus_path = Path('corpora/kubhist2-carlscronaswekoblad-1780.parquet')
corpus = ParquetCorpus(str(corpus_path))
all_usages = corpus.search(SearchTerm(lemma='man'))['lemma=man']
usages = all_usages.sample(100, random_seed=42)

if args.reuse_run:
    with jsonlines.open(args.reuse_run / 'sample/lemma=man_usages.jsonl') as reader:
        previous = list(reader)
    assert [r['text'] for r in previous] == [u.text() for u in usages]
    assert [r['offsets'] for r in previous] == [u.offsets for u in usages]
    assert [r['time'] for r in previous] == [str(u.time) for u in usages]
    embeddings = np.load(args.reuse_run / 'usage_embeddings.npy', allow_pickle=False)
else:
    from languagechange.models.representation.contextualized import XL_LEXEME
    model = XL_LEXEME(device='cpu')
    embeddings = model.encode(usages, batch_size=2)
assert embeddings.shape == (len(usages), 1024)
assert np.isfinite(embeddings).all()

# Keep embedding groups and labels in the same chronological order.
usage_years = [_parse_year(u.time) for u in usages]
time_labels = sorted(set(usage_years))
embs_list = [embeddings[np.array([year == label for year in usage_years])] for label in time_labels]
ts_apd = TimeSeries(embs_list, change_metric=APD(), timeseries_type='consecutive', time_labels=time_labels)
clustering_algo = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='complete',
)
ts_jsd_1 = TimeSeries(
    embs_list, change_metric=JSD(), clustering_algorithm=clustering_algo,
    timeseries_type='consecutive', time_labels=time_labels,
)
labels = Clustering(clustering_algo).get_cluster_results(np.concatenate(embs_list)).labels
splits = np.cumsum([len(group) for group in embs_list][:-1])
cluster_labels = np.split(labels, splits)
ts_jsd_2 = TimeSeries(cluster_labels, change_metric=JSD(), timeseries_type='consecutive', time_labels=time_labels)
assert np.allclose(ts_jsd_1.series, ts_jsd_2.series)

series = {'apd_scores': ts_apd, 'jsd_from_embeddings': ts_jsd_1, 'jsd_from_labels': ts_jsd_2}
exports = {}
for name, ts in series.items():
    records = [
        {'from_year': str(start), 'to_year': str(end), 'score': float(score)}
        for start, end, score in zip(time_labels[:-1], time_labels[1:], ts.series)
    ]
    exports[name + '.jsonl'] = records
    ts.plot(
        xlabel='Later year of comparison',
        ylabel='Average pairwise distance' if name == 'apd_scores' else 'JSD score',
        save_f=str(output_dir / (name + '.png')),
    )
    plt.close('all')
exports['cluster_labels.jsonl'] = [
    {'year': str(year), 'labels': group.tolist()}
    for year, group in zip(time_labels, cluster_labels)
]
for filename, records in exports.items():
    with jsonlines.open(output_dir / filename, mode='w') as writer:
        writer.write_all(records)
    with jsonlines.open(output_dir / filename) as reader:
        assert list(reader) == records

UsageDictionary({'lemma=man': usages}).save(str(output_dir / 'sample'))
np.save(output_dir / 'usage_embeddings.npy', embeddings)
assert np.array_equal(np.load(output_dir / 'usage_embeddings.npy', allow_pickle=False), embeddings)
with jsonlines.open(output_dir / 'sample/lemma=man_usages.jsonl') as reader:
    saved = list(reader)
assert [r['text'] for r in saved] == [u.text() for u in usages]
assert [r['offsets'] for r in saved] == [u.offsets for u in usages]
assert [r['time'] for r in saved] == [str(u.time) for u in usages]
metadata = {
    'embedding_model': 'pierluigic/xl-lexeme',
    'embeddings_source': 'saved sample-aligned embeddings' if args.reuse_run else 'XL_LEXEME.encode',
    'device': 'cpu', 'batch_size': 2, 'sample_seed': 42,
    'corpus_sha256': hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    'embedding_sha256': hashlib.sha256((output_dir / 'usage_embeddings.npy').read_bytes()).hexdigest(),
    'embedding_shape': list(embeddings.shape),
    'years': [str(year) for year in time_labels], 'year_counts': [len(g) for g in embs_list],
    'clustering': {'metric': 'cosine', 'linkage': 'complete', 'distance_threshold': 0.5, 'n_clusters': None},
    'cluster_count': len(np.unique(labels)),
    'jsd_approaches_match': True,
    'runtime': {name: version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers']},
}
(output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Sample size:', len(usages))
print('Year order:', metadata['years'])
print('Group sizes:', metadata['year_counts'])
print('Number of clusters:', metadata['cluster_count'])
print('APD scores:', ts_apd.series.tolist())
print('JSD scores:', ts_jsd_1.series.tolist())
print('JSD results match:', metadata['jsd_approaches_match'])
print('Execution succeeded.')
print('Output folder:', output_dir.resolve())
