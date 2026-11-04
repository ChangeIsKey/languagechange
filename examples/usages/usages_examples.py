# Example usage of the classes in languagechange.usages
from languagechange.search import SearchTerm
from languagechange.corpora import ParquetCorpus
from pathlib import Path

from languagechange.usages import TargetUsage, UsageDictionary

# A TargetUsage consists of a text and the character offsets specifying where the target word is in the text.
target_usage = TargetUsage("I have a dog", offsets=[9, 12])
dog_target = target_usage.text()[target_usage.start(): target_usage.end()]

# Let's get some more usages to work with from a dataset.
corpus = ParquetCorpus("corpora/kubhist2-carlscronaswekoblad-1780.parquet")
usages = corpus.search(SearchTerm(lemma="man"))  # We get back a UsageDictionary of TargetUsageList

# Choose the usages of the given search term
man_usages = usages["lemma=man"]

# This class has methods for grouping by time or time interval as well as sampling usages.
# By default, grouping is done per year.
grouped = man_usages.group_by_time()
first_usage_1780 = next(iter(grouped.values()))[0]

# Group by exact time values instead:
grouped_per_date = man_usages.group_by_time(use_year=False)

intervals = [["1780", "1781"], ["1782", "1784"]]
grouped_per_interval = man_usages.group_by_interval(intervals, use_year=True)
first_usage_1780_1781 = next(iter(grouped_per_interval.values()))[0]

# Sample 10 usages randomly
sampled = man_usages.sample(10, random_seed=42)

# We can also sample N usages per time period or time interval:
sampled_per_year = man_usages.sample_per_time(n_samples=10, random_seed=42)

# Save actual usage records, one JSON object per line.
output_dir = Path("outputs/usages_native")
usages.save(str(output_dir))
UsageDictionary({"lemma=man": sampled}).save(str(output_dir / "sample"))
UsageDictionary({"lemma=man": sampled_per_year}).save(str(output_dir / "sample_per_year"))
grouped.save(str(output_dir / "by_year"))
grouped_per_interval.save(str(output_dir / "by_interval"))

print("Target:", dog_target)
print("Total usages:", len(man_usages))
print("Year groups:", [(str(year), len(items)) for year, items in grouped.items()])
print("Date groups:", len(grouped_per_date))
print("Year intervals:", [(str(interval), len(items)) for interval, items in grouped_per_interval.items()])
print("Sample size:", len(sampled))
print("Yearly sample size:", len(sampled_per_year))
print("JSONL directory:", output_dir)
