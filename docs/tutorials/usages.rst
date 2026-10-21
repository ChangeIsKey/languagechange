Working with Word Usages
========================

Introduction
------------

A word usage contains a text and the character positions of a target word.
It can also contain a date and other metadata. This tutorial follows
``usages_examples.py``: identify a target, find its corpus usages, group
and sample them, and save the resulting records as JSONL.

.. contents:: In this tutorial
   :local:
   :depth: 2

Prerequisites
-------------

Install ``languagechange`` in your Python environment. Download
:download:`the example Parquet corpus
<data/kubhist2-carlscronaswekoblad-1780.parquet>` and put it in a folder
named ``corpora`` inside your working directory. The expected path is
``corpora/kubhist2-carlscronaswekoblad-1780.parquet``.

The corpus is from Språkbanken and is distributed under CC BY 4.0.
See :doc:`corpora` for the source citation and how the file was prepared.
Run the Python blocks below in order in the same session.

You can also download :download:`usages_examples.py
<../../examples/usages/usages_examples.py>` into your working directory
and run the complete example from a terminal with your environment active:

.. code-block:: console

   python usages_examples.py

Identifying a target in context
-------------------------------

Character positions start at zero. The end position is excluded, as in a
Python string slice. Here, positions 9 through 11 identify ``dog``:

.. code-block:: python

   from languagechange.usages import TargetUsage

   target_usage = TargetUsage("I have a dog", offsets=[9, 12])
   dog_target = target_usage.text()[target_usage.start():target_usage.end()]
   print(target_usage.text())
   print(dog_target)

Printed output:

.. code-block:: text

   I have a dog
   dog

This example has no date. The corpus usages below include dates, so we can
group them by time.

Finding and saving corpus usages
---------------------------------

Search the corpus for the lemma ``man``. The result is a ``UsageDictionary``:
each search-term key maps to a list of matching usages.

.. code-block:: python

   from languagechange.corpora import ParquetCorpus
   from languagechange.search import SearchTerm

   corpus = ParquetCorpus("corpora/kubhist2-carlscronaswekoblad-1780.parquet")
   usages = corpus.search(SearchTerm(lemma="man"))
   man_usages = usages["lemma=man"]
   print(len(man_usages))

Printed output:

.. code-block:: text

   480

Use ``UsageDictionary.save()`` to write the actual usage records. The writer
creates the output folder if it does not exist:

.. code-block:: python

   from pathlib import Path

   output_dir = Path("outputs/usages_native")
   usages.save(str(output_dir))

This creates ``outputs/usages_native/lemma=man_usages.jsonl`` with 480
records. JSONL contains one JSON object per line. The first three are:

.. literalinclude:: outputs/usages_outputs.jsonl
   :language: json
   :lines: 1-3

Each record retains the text, offsets, full date, token, lemma,
part-of-speech, and search-target metadata supplied by the corpus. These
records have no ``id`` field. Historical spelling is preserved.

:download:`Download all 480 usages <outputs/usages_outputs.jsonl>`.

Grouping by year
----------------

By default, ``group_by_time()`` groups usages by year. Its keys are time
objects; convert them to strings when printing them:

.. code-block:: python

   grouped = man_usages.group_by_time()
   print([(str(year), len(items)) for year, items in grouped.items()])
   first_usage_1780 = next(iter(grouped.values()))[0]
   print(first_usage_1780.time)

Printed output:

.. code-block:: text

   [('1780', 76), ('1781', 114), ('1782', 281), ('1784', 9)]
   1780-01-08

The counts sum to 480. Grouping preserves each usage's original full date.
To save one JSONL file for each year:

.. code-block:: python

   grouped.save(str(output_dir / "by_year"))

For example, the 76 usages from 1780 are written to
``outputs/usages_native/by_year/1780_usages.jsonl``.

Grouping by exact date
----------------------

Pass ``use_year=False`` to group by individual dates instead of years:

.. code-block:: python

   grouped_per_date = man_usages.group_by_time(use_year=False)
   print("Date groups:", len(grouped_per_date))
   print([(str(date), len(items)) for date, items in list(grouped_per_date.items())[:5]])
   print("Total usages:", sum(len(items) for items in grouped_per_date.values()))

Printed output:

.. code-block:: text

   Date groups: 129
   [('1780-01-08', 11), ('1780-01-15', 1), ('1780-01-22', 1), ('1780-01-29', 2), ('1780-02-19', 2)]
   Total usages: 480

Grouping by year intervals
--------------------------

Pass ``use_year=True`` when your boundaries are complete calendar years.
This includes every date in both the starting and ending years:

.. code-block:: python

   intervals = [["1780", "1781"], ["1782", "1784"]]
   grouped_per_interval = man_usages.group_by_interval(intervals, use_year=True)
   print([(str(interval), len(items)) for interval, items in grouped_per_interval.items()])
   print("Total in intervals:", sum(len(items) for items in grouped_per_interval.values()))
   first_usage_1780_1781 = next(iter(grouped_per_interval.values()))[0]
   print(first_usage_1780_1781.time)

Printed output:

.. code-block:: text

   [('1780 - 1781', 190), ('1782 - 1784', 290)]
   Total in intervals: 480
   1780-01-08

The first interval contains 76 usages from 1780 and 114 from 1781. The
second contains 281 from 1782 and nine from 1784. There are no 1783 usages
in this search result.

Save the records in each interval:

.. code-block:: python

   grouped_per_interval.save(str(output_dir / "by_interval"))

The files are ``by_interval/1780 - 1781_usages.jsonl`` and
``by_interval/1782 - 1784_usages.jsonl`` inside ``outputs/usages_native``.
Download the :download:`190 usages from 1780–1781
<outputs/usages_interval_1780_1781_outputs.jsonl>` or the
:download:`290 usages from 1782–1784
<outputs/usages_interval_1782_1784_outputs.jsonl>`.

Sampling ten usages
-------------------

Sampling selects usages without changing the full list. A fixed random
seed repeats the selection for the same input order and sampling environment:

.. code-block:: python

   sampled = man_usages.sample(10, random_seed=42)
   print("Sample size:", len(sampled))

Printed output:

.. code-block:: text

   Sample size: 10

Save the selected objects through the same writer:

.. code-block:: python

   from languagechange.usages import UsageDictionary

   UsageDictionary({"lemma=man": sampled}).save(str(output_dir / "sample"))

This creates ``outputs/usages_native/sample/lemma=man_usages.jsonl``.
The first two records are:

.. literalinclude:: outputs/usages_sample_outputs.jsonl
   :language: json
   :lines: 1-2

:download:`Download the ten sampled usages <outputs/usages_sample_outputs.jsonl>`.

Sampling per year
-----------------

``sample_per_time()`` selects up to the requested number from each year.
It does not duplicate usages to fill a small group:

.. code-block:: python

   sampled_per_year = man_usages.sample_per_time(n_samples=10, random_seed=42)
   print("Sampled usages:", len(sampled_per_year))
   print([(str(year), len(items)) for year, items in sampled_per_year.group_by_time().items()])

Printed output:

.. code-block:: text

   Sampled usages: 39
   [('1780', 10), ('1781', 10), ('1782', 10), ('1784', 9)]

The final year contributes nine usages because only nine are available.
Save this sample in its own folder:

.. code-block:: python

   UsageDictionary({"lemma=man": sampled_per_year}).save(str(output_dir / "sample_per_year"))

The first two records of
``outputs/usages_native/sample_per_year/lemma=man_usages.jsonl`` are:

.. literalinclude:: outputs/usages_year_sample_outputs.jsonl
   :language: json
   :lines: 1-2

:download:`Download the 39 yearly sampled usages
<outputs/usages_year_sample_outputs.jsonl>`.

Checking saved records
----------------------

Read a file back to check that it is valid JSONL and preserves the usage
data. Here we check the full export:

.. code-block:: python

   import jsonlines

   with jsonlines.open(output_dir / "lemma=man_usages.jsonl") as reader:
       saved_usages = list(reader)
   print("Saved usages:", len(saved_usages))
   print("Text matches:", [r["text"] for r in saved_usages] == [u.text() for u in man_usages])
   print("Offsets match:", [r["offsets"] for r in saved_usages] == [u.offsets for u in man_usages])
   print("Dates match:", [r["time"] for r in saved_usages] == [str(u.time) for u in man_usages])

Printed output:

.. code-block:: text

   Saved usages: 480
   Text matches: True
   Offsets match: True
   Dates match: True

You can repeat these checks for a sample or interval by using its saved
file and the corresponding list of usages.

Outputs saved as JSONL
----------------------

The files below are copies of the records written by
``UsageDictionary.save()``. Each line is one usage object. Counts and other
printed messages are shown separately beside the relevant code.

All usages, first three records:

.. literalinclude:: outputs/usages_outputs.jsonl
   :language: json
   :lines: 1-3

Ten-record sample, first two records:

.. literalinclude:: outputs/usages_sample_outputs.jsonl
   :language: json
   :lines: 1-2

Yearly sample, first two records:

.. literalinclude:: outputs/usages_year_sample_outputs.jsonl
   :language: json
   :lines: 1-2

Download the complete records:

* :download:`All usages: 480 records <outputs/usages_outputs.jsonl>`.
* :download:`Random sample: 10 records <outputs/usages_sample_outputs.jsonl>`.
* :download:`Yearly sample: 39 records <outputs/usages_year_sample_outputs.jsonl>`.
* :download:`1780–1781 interval: 190 records
  <outputs/usages_interval_1780_1781_outputs.jsonl>`.
* :download:`1782–1784 interval: 290 records
  <outputs/usages_interval_1782_1784_outputs.jsonl>`.

Running ``usages_examples.py`` saves the full list in
``outputs/usages_native/``. The ``sample/``, ``sample_per_year/``,
``by_year/``, and ``by_interval/`` subfolders keep each selection separate.

Summary
-------

You can identify a target using character offsets, find its corpus usages,
group them by year, date, or complete-year intervals, and sample them.
Use a fixed seed for repeatable sampling. Save the actual usage records
as JSONL and read them back to check that their text, offsets, and dates
are preserved.
