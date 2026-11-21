"""Run the corpus example: Parquet search, sampling and export workflow."""
import argparse
from pathlib import Path

from languagechange.corpora import ParquetCorpus
from languagechange.search import SearchTerm
from languagechange.usages import UsageDictionary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--corpus", type=Path, default=Path("corpora/kubhist2-carlscronaswekoblad-1780.parquet"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/corpora_native"))
args = parser.parse_args()

corpus = ParquetCorpus(str(args.corpus))
results = corpus.search(SearchTerm(lemma="man"))
man_usages = results["lemma=man"]
if not man_usages:
    raise ValueError("The example corpus returned no usages for lemma=man.")
first = man_usages[0]
print("Usages:", len(man_usages))
print("First target:", first.text()[first.start():first.end()])
results.save(str(args.output_dir))
sample = man_usages.sample(10, random_seed=42)
UsageDictionary({"lemma=man": sample}).save(str(args.output_dir / "sample"))
grouped = man_usages.group_by_time()
grouped.save(str(args.output_dir / "by_year"))
print("Sample:", len(sample))
print("Year groups:", [(str(year), len(items)) for year, items in grouped.items()])
print("Output folder:", args.output_dir.resolve())
