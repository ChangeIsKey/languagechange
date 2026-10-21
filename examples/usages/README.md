# Word usage example

This example finds word usages, groups them by year or date, samples them,
and saves the actual records as JSONL.

Install `languagechange` and place
`kubhist2-carlscronaswekoblad-1780.parquet` in a `corpora` folder in your
working directory. The Parquet input is prepared in the corpus tutorial.
The source is [Språkbanken's Kubhist 2: Carlscronas Wekoblad 1780-talet](https://doi.org/10.23695/4qx9-j028),
available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

From that working directory, run:

```bash
python /path/to/languagechange/examples/usages/usages_examples.py
```

The example uses `use_year=True` to include the full years in each interval.
The intervals contain 190 and 290 usages. Sampling uses `random_seed=42`
for repeatable selections with the same inputs and environment.

Results are saved in `outputs/usages_native/`:

- `lemma=man_usages.jsonl`: 480 usages.
- `sample/lemma=man_usages.jsonl`: 10 sampled usages.
- `sample_per_year/lemma=man_usages.jsonl`: 39 usages, up to ten per year.
- `by_year/`: one file for each of the four years.
- `by_interval/`: one file for each of the two intervals.

Each JSONL line is an actual usage record saved by `UsageDictionary.save()`.
Counts and the output folder are printed in the terminal.
