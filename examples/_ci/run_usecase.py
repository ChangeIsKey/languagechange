"""Execute one tutorial example in a fresh directory and validate its outputs.

No saved predictions or embedding snapshots are used as execution substitutes.
Model weights may be cached by Hugging Face; example result caches are local
to the fresh run directory. This file never downloads or executes code from
a dataset: it runs the fixed example paths listed below.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    'corpora': 'examples/corpora/corpora_examples.py',
    'usages': 'examples/usages/usages_examples.py',
    'benchmarks': 'examples/benchmarks/benchmark_examples.py',
    'definitions': 'examples/definitions/definition_t5_xl_examples.py',
    'prompting': 'examples/prompting/prompting_groq_examples.py',
    'timeseries': 'examples/timeseries/timeseries_examples.py',
    'visualization': 'examples/visualizations/visualization_examples.py',
}
SECRET_NAMES = ('GROQ_API_KEY', 'LANGSMITH_API_KEY')


def in_github_actions():
    return os.environ.get('GITHUB_ACTIONS') == 'true'


def ci_sample(usages):
    from languagechange.usages import TargetUsageList
    grouped = usages.group_by_time()
    return TargetUsageList([u for i, year in enumerate(sorted(grouped))
                           for u in grouped[year].sample(3 if i < 2 else 2, random_seed=42)])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    def reject(value):
        raise ValueError(f'Nonstandard JSON constant {value} in {path.name}')
    with path.open(encoding='utf-8') as stream:
        result = [json.loads(line, parse_constant=reject) for line in stream]
    require(bool(result), f'Empty result file: {path}')
    require(all(isinstance(row, dict) for row in result), f'Expected JSON objects in {path}')
    return result


def check_prerequisites(topic):
    if topic == 'prompting':
        missing = [name for name in SECRET_NAMES if not os.environ.get(name, '').strip()]
        require(not missing, 'Configure these GitHub Actions secrets before prompting: ' + ', '.join(missing))
    if topic == 'definitions' and not in_github_actions():
        import torch
        require(torch.cuda.is_available(), 'T5 XL requires an available CUDA GPU; no CPU or smaller-model fallback is used.')
        free, _ = torch.cuda.mem_get_info()
        require(free >= 13 * 1024**3, 'T5 XL requires at least 13 GiB of free GPU memory.')


def prepare_corpus(work):
    manifest = json.loads((ROOT / 'examples/_ci/data_manifest.json').read_text())['corpus']
    source = ROOT / manifest['path']
    require(source.is_file(), f'Missing committed corpus: {source}')
    require(hashlib.sha256(source.read_bytes()).hexdigest() == manifest['sha256'],
            'The input corpus checksum changed. Review the fixture and update its manifest deliberately.')
    destination = work / 'corpora' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return manifest


def commands(topic, output):
    script = str(ROOT / SCRIPTS[topic])
    if topic == 'definitions':
        return [
            [sys.executable, script, 'generate', '--download-dataset', '--output-dir', str(output)],
            [sys.executable, script, 'encode', '--output-dir', str(output)],
        ]
    return [[sys.executable, script, '--output-dir', str(output)]]


def run_command(command, work, logfile):
    environment = os.environ.copy()
    environment.update(MPLBACKEND='Agg', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    if in_github_actions():
        environment['CUDA_VISIBLE_DEVICES'] = ''
    environment.setdefault('OMP_NUM_THREADS', '2')
    environment.setdefault('MKL_NUM_THREADS', '2')
    # Redact known credentials from persisted logs as well as console output.
    secrets = [environment[name] for name in SECRET_NAMES if environment.get(name)]
    with logfile.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=work, env=environment, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                for secret in secrets:
                    line = line.replace(secret, '[REDACTED]')
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def check_png(path):
    from PIL import Image
    with Image.open(path) as image:
        require(image.format == 'PNG' and min(image.size) > 0, f'Invalid graph: {path}')
        image.verify()


def check_array(path, shape):
    import numpy as np
    array = np.load(path, allow_pickle=False)
    require(array.shape == shape and np.isfinite(array).all(), f'Invalid embedding array: {path}')
    return array


def check_usage_records(path, expected):
    records = read_jsonl(path)
    require(len(records) == len(expected), f'Usage count mismatch: {path}')
    for row, usage in zip(records, expected):
        require(row.get('text') == usage.text(), f'Usage text/order mismatch: {path}')
        require(row.get('offsets') == list(usage.offsets), f'Target offsets mismatch: {path}')
        require(row.get('time') == str(usage.time), f'Usage date mismatch: {path}')
        for field in ('token', 'lemma', 'pos_tag', 'target'):
            if hasattr(usage, field):
                require(row.get(field) == getattr(usage, field), f'{field} metadata mismatch: {path}')
    return records


def source_usages(work):
    from languagechange.corpora import ParquetCorpus
    from languagechange.search import SearchTerm
    return ParquetCorpus(str(work / 'corpora/kubhist2-carlscronaswekoblad-1780.parquet')).search(
        SearchTerm(lemma='man'))['lemma=man']


def validate_outputs(topic, output, work, manifest):
    import numpy as np
    if topic in ('corpora', 'usages', 'timeseries', 'visualization'):
        usages = source_usages(work)
        require(len(usages) == manifest['expected_usages'], 'Unexpected corpus search result count.')
    if topic in ('corpora', 'usages'):
        check_usage_records(output / 'lemma=man_usages.jsonl', usages)
        check_usage_records(output / 'sample/lemma=man_usages.jsonl', usages.sample(10, random_seed=42))
        grouped = usages.group_by_time()
        counts = {str(year): len(items) for year, items in grouped.items()}
        require(counts == manifest['expected_year_counts'], 'Year groups changed.')
        for year, items in grouped.items():
            check_usage_records(output / f'by_year/{year}_usages.jsonl', items)
        if topic == 'usages':
            check_usage_records(output / 'sample_per_year/lemma=man_usages.jsonl',
                                usages.sample_per_time(n_samples=10, random_seed=42))
            intervals = usages.group_by_interval([['1780', '1781'], ['1782', '1784']], use_year=True)
            require(sorted(map(len, intervals.values())) == [190, 290], 'Full-year interval counts changed.')
            for interval, items in intervals.items():
                check_usage_records(output / f'by_interval/{interval}_usages.jsonl', items)
        return {'usages': len(usages), 'year_counts': counts}
    if topic == 'prompting':
        durel = read_jsonl(output / 'durel_predictions.jsonl')
        wic = read_jsonl(output / 'wic_predictions.jsonl')
        require(len(durel) == len(wic) == 2, 'Expected two responses for each prompting schema.')
        require(all(set(r) == {'durel'} and type(r['durel']) is int and 1 <= r['durel'] <= 4 for r in durel),
                'Invalid DURel schema or range.')
        require(all(set(r) == {'wic_label'} and type(r['wic_label']) is bool for r in wic),
                'Invalid WiC schema. Boolean labels are required.')
        return {'durel_responses': len(durel), 'wic_responses': len(wic)}
    if topic == 'definitions':
        rows = read_jsonl(output / 'definition_predictions.jsonl')
        require(len(rows) == 10, 'Expected ten newly generated plane definitions.')
        require(all(r.get('Targets') == 'plane' and isinstance(r.get('Generated_Definition'), str)
                    and r['Generated_Definition'].strip() for r in rows), 'Invalid definition records.')
        check_array(output / 'definition_embeddings.npy', (10, 768))
        return {'definitions': len(rows), 'embedding_shape': [10, 768], 'model': 'ltg/flan-t5-definition-en-xl'}
    if topic == 'benchmarks':
        report = json.loads((output / 'benchmark_results.json').read_text())
        require(report.get('ci_mode') == in_github_actions(), 'Benchmark execution mode mismatch.')
        for key in ('wic_splits', 'wsd_splits', 'wsd_wic_splits'):
            require(set(report[key]) == {'train', 'dev', 'test'} and all(v > 0 for v in report[key].values()),
                    f'Invalid benchmark split counts: {key}')
        require(report['custom_wic_pairs'] == 2 and report['custom_wsi_usages'] == 3, 'Custom benchmark examples changed.')
        for key in ('wic_f1', 'wsd_f1', 'spearman_statistic', 'spearman_pvalue'):
            require(math.isfinite(report[key]), f'Non-finite benchmark score: {key}')
        require(0 <= report['wic_f1'] <= 1 and report['wsd_f1'] == 1.0, 'Invalid benchmark evaluation.')
        require(report['wsi_ari'] and all(v == 1.0 for v in report['wsi_ari'].values()), 'Gold-label ARI check failed.')
        require(report['wsi_purity'] and all(v == 1.0 for v in report['wsi_purity'].values()), 'Gold-label purity check failed.')
        for name in ('egg_judgments.csv', 'egg_clusters.csv', 'egg_combined_judgments.csv', 'egg_combined_clusters.csv'):
            require((output / name).is_file() and (output / name).stat().st_size > 0, f'Missing benchmark output: {name}')
        if in_github_actions():
            for key in ('wic_splits', 'wsd_splits'):
                require(all(v <= 10 for v in report[key].values()), 'CI benchmark splits exceed ten examples.')
            for name in ('egg_judgments.csv', 'egg_combined_judgments.csv'):
                with (output / name).open() as stream:
                    judgments = list(csv.DictReader(stream, delimiter='\t'))
                identifiers = {row[key] for row in judgments for key in ('identifier1', 'identifier2')}
                require(len(identifiers) == 10 and len(judgments) == 45,
                        'Expected all 45 pairwise judgments for ten CI usages.')
        check_png(output / 'leilighet_graph.png')
        return report
    if topic == 'visualization':
        if in_github_actions():
            usages = ci_sample(usages)
        grouped = usages.group_by_time()
        expected = [u for year in sorted(grouped) for u in grouped[year]]
        check_usage_records(output / 'usages/lemma=man_usages.jsonl', expected)
        check_array(output / 'usage_embeddings.npy', (len(expected), 1024))
        check_array(output / 'reduced_embeddings.npy', (len(expected), 2))
        rows = read_jsonl(output / 'cluster_labels.jsonl')
        require(len(rows) == len(expected), 'One cluster label is required per visualization usage.')
        for i, (row, usage) in enumerate(zip(rows, expected)):
            require(row.get('usage_index') == i and row.get('year') == str(usage.time)[:4]
                    and type(row.get('cluster')) is int, 'Visualization labels are not aligned with usages.')
        check_png(output / 'usage_embeddings_by_year.png')
        return {'usages': len(expected), 'clusters': len({row['cluster'] for row in rows})}
    if topic == 'timeseries':
        from scipy.spatial.distance import cdist, jensenshannon
        sample = ci_sample(usages) if in_github_actions() else usages.sample(100, random_seed=42)
        check_usage_records(output / 'sample/lemma=man_usages.jsonl', sample)
        vectors = check_array(output / 'usage_embeddings.npy', (len(sample), 1024))
        years = sorted({str(u.time)[:4] for u in sample})
        groups = [vectors[np.array([str(u.time).startswith(year) for u in sample])] for year in years]
        labels = read_jsonl(output / 'cluster_labels.jsonl')
        require([r['year'] for r in labels] == years, 'Cluster years are misordered.')
        for row, group in zip(labels, groups):
            require(len(row['labels']) == len(group) and all(type(v) is int for v in row['labels']), 'Invalid cluster-label groups.')
        apd = read_jsonl(output / 'apd_scores.jsonl')
        jsd = read_jsonl(output / 'jsd_from_embeddings.jsonl')
        require(jsd == read_jsonl(output / 'jsd_from_labels.jsonl'), 'The two JSD approaches disagree.')
        for rows in (apd, jsd):
            require(len(rows) == len(years) - 1, 'Incorrect number of consecutive scores.')
            require([(r['from_year'], r['to_year']) for r in rows] == list(zip(years[:-1], years[1:])), 'Score years are misaligned.')
            require(all(type(r['score']) in (int, float) and math.isfinite(r['score']) for r in rows), 'Invalid change scores.')
        actual_apd = [float(cdist(a, b, metric='cosine').mean()) for a, b in zip(groups[:-1], groups[1:])]
        require(np.allclose(actual_apd, [r['score'] for r in apd], rtol=1e-7, atol=1e-9), 'APD does not match saved vectors.')
        keys = sorted({v for row in labels for v in row['labels']})
        counts = [Counter(row['labels']) for row in labels]
        actual_jsd = [float(jensenshannon([a[k] for k in keys], [b[k] for k in keys])) for a, b in zip(counts[:-1], counts[1:])]
        require(np.allclose(actual_jsd, [r['score'] for r in jsd], rtol=1e-7, atol=1e-9), 'JSD does not match saved cluster distributions.')
        for name in ('apd_scores.png', 'jsd_from_embeddings.png', 'jsd_from_labels.png'):
            check_png(output / name)
        return {'sample_size': len(sample), 'year_counts': [len(g) for g in groups], 'score_count': len(apd)}
    raise ValueError(f'Unknown use case: {topic}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usecase', choices=sorted(SCRIPTS))
    parser.add_argument('--output-dir', type=Path, required=True, help='Use a new or empty directory for each run.')
    args = parser.parse_args(argv)
    destination = args.output_dir.resolve()
    require(not destination.exists() or not any(destination.iterdir()), 'Choose a new or empty output directory; old results must not pass as a new execution.')
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / 'work'
    work.mkdir()
    results = work / 'results'
    report = {'usecase': args.usecase, 'status': 'failed', 'started_at_unix': time.time(),
              'example': SCRIPTS[args.usecase], 'commit': os.environ.get('GITHUB_SHA'),
              'ci_mode': in_github_actions()}
    try:
        check_prerequisites(args.usecase)
        manifest = prepare_corpus(work)
        report['input_sha256'] = manifest['sha256']
        report['script_sha256'] = hashlib.sha256((ROOT / SCRIPTS[args.usecase]).read_bytes()).hexdigest()
        for stage, command in enumerate(commands(args.usecase, results), start=1):
            print(f'Running {args.usecase}, stage {stage}', flush=True)
            run_command(command, work, destination / f'stage-{stage}.log')
        report['checks'] = validate_outputs(args.usecase, results, work, manifest)
        report['status'] = 'passed'
        returncode = 0
    except (Exception, SystemExit) as error:
        report['error_type'] = type(error).__name__
        # Prerequisite messages contain names, never credential values.
        message = str(error)
        for name in SECRET_NAMES:
            if os.environ.get(name):
                message = message.replace(os.environ[name], '[REDACTED]')
        report['error'] = message
        print(f'Use case failed: {message}', file=sys.stderr, flush=True)
        returncode = 1
    report['finished_at_unix'] = time.time()
    (destination / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"{args.usecase}: {report['status']}; report: {destination / 'run.json'}", flush=True)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
