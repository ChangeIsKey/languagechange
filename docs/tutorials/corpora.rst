Working with Corpora
====================

Introduction
------------

A corpus is a collection of texts that can be used to study language.
The ``languagechange`` package provides tools for loading, processing,
and searching corpora.

In this tutorial, we will learn how to:

* Load a corpus.
* Search for words and lemmas.
* Explore word usages.
* Group and sample usages for further analysis.

.. contents:: In this tutorial
   :local:

Prerequisites
-------------

Before starting, make sure that you have installed ``languagechange``.
Run the examples in order in the same Python session or notebook.

Loading a corpus
----------------

We will use *Kubhist 2: Carlscronas Wekoblad 1780-talet*, a corpus of historical
Swedish newspaper text distributed by Språkbanken. It already contains tokens,
lemmas, part-of-speech tags, sentence identifiers, and dates.

Download :download:`the example Parquet corpus
<data/kubhist2-carlscronaswekoblad-1780.parquet>` (about 1.9 MB) and save it as
``kubhist2-carlscronaswekoblad-1780.parquet`` in your working directory. In a
notebook, this is normally the directory containing the notebook. Then load it:

.. code-block:: python

   from languagechange.corpora import ParquetCorpus

   corpus = ParquetCorpus("kubhist2-carlscronaswekoblad-1780.parquet")

The file is a Parquet conversion of the XML corpus from
`Språkbanken's resource page
<https://sprakbanken.se/resurser/kubhist2-carlscronaswekoblad-1780>`_.
Source: Språkbanken (2024), *Kubhist 2: Carlscronas Wekoblad 1780-talet*,
updated 2024-01-06, `doi:10.23695/4qx9-j028
<https://doi.org/10.23695/4qx9-j028>`_. The data is available under
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_. The conversion
changes the file format and retains the text and annotations, including
historical spellings and OCR errors.

Starting from XML instead
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also download the compressed XML file directly from Språkbanken and
convert it yourself. This is an alternative to loading the supplied Parquet
file above; you do not need to run both examples.

.. code-block:: python

   from urllib.request import urlretrieve
   from languagechange.corpora import SprakBankenCorpus

   xml_filename = "kubhist2-carlscronaswekoblad-1780.xml.bz2"
   urlretrieve(
       "https://spraakbanken.gu.se/resurser/meningsmangder/" + xml_filename,
       xml_filename,
   )
   xml_corpus = SprakBankenCorpus(xml_filename)
   corpus = xml_corpus.cast_to_parquet()

``SprakBankenCorpus`` reads the ``.xml.bz2`` file without a separate
decompression step. The conversion writes
``kubhist2-carlscronaswekoblad-1780.parquet`` in the working directory and
returns a ``ParquetCorpus``. It replaces a file of the same name if one already
exists. Downloading and converting may take a little time.

Searching and inspecting usages
-------------------------------

Searching for a word
~~~~~~~~~~~~~~~~~~~~

We can search the corpus for usages associated with a particular lemma.
For example, to search for the lemma ``man``:

.. code-block:: python

   from languagechange.search import SearchTerm

   results = corpus.search(SearchTerm(lemma="man"))

The result contains the usages matching the search. We can retrieve the
group corresponding to our search term:

.. code-block:: python

   man_usages = results["lemma=man"]

The key ``"lemma=man"`` describes the search we performed. A lemma search can
match different surface forms; below, the first matching token is ``Man``.

Exploring word usages
~~~~~~~~~~~~~~~~~~~~~

The search returns 480 usages for this corpus and query. To inspect the
first sentence and its matched token:

.. code-block:: python

   print(len(man_usages))
   first_usage = man_usages[0]
   print(first_usage.text())
   print(first_usage.text()[first_usage.start():first_usage.end()])

Printed output:

.. code-block:: text

   480
   Am:ts Sjukhuset döde 37 Man , samt 2tne pä wädeligit sätt omkomne ditförde och jordfäste .
   Man

The ``start()`` and ``end()`` methods give character offsets into the
sentence. The end offset is exclusive. Printed counts and sentences are
diagnostics; the JSONL files below contain the exported data records.

Saving and organizing usages
----------------------------

Saving the search results as JSONL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the search result's ``save()`` method. This is the library's
``UsageDictionary.save()`` exporter; it writes one usage object per line.
It does not wrap terminal output in an extra section or output field.

.. code-block:: python

   results.save("corpora_outputs")

This creates ``corpora_outputs/lemma=man_usages.jsonl`` with all 480 usages.
The first three records from that file are:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 1-3

Each record contains the sentence in ``text``, target character ``offsets``,
and the source ``time``. This corpus also supplies token, lemma,
part-of-speech, and search-target metadata, which the exporter preserves.
The exported usages here have no ``id`` field; no ID is added manually.
Dates retain their actual full source values rather than being replaced
with years from another example. Historical spellings remain as supplied.

Download :download:`all 480 exported usage records
<outputs/corpora_outputs.jsonl>`. The downloadable file is a byte-for-byte
copy of the library-created file, renamed for this documentation page.

Sampling usages
~~~~~~~~~~~~~~~

Select ten usages with a fixed seed and export them through the same
library method. Constructing a ``UsageDictionary`` lets a selected list of
usages use the native writer as well:

.. code-block:: python

   from languagechange.usages import UsageDictionary

   sample = man_usages.sample(10, random_seed=42)
   UsageDictionary({"lemma=man": sample}).save("corpora_outputs/sample")
   print(len(sample))

Printed output:

.. code-block:: text

   10

The file ``corpora_outputs/sample/lemma=man_usages.jsonl`` contains ten
usage records. Its first two records are:

.. literalinclude:: outputs/corpora_sample_outputs.jsonl
   :language: json
   :lines: 1-2

Download :download:`the ten sampled records
<outputs/corpora_sample_outputs.jsonl>`. The seed makes sampling
repeatable for the same input order and NumPy version.

Grouping usages by time
~~~~~~~~~~~~~~~~~~~~~~~

When usages contain time information, group them by year and export each
group. The grouping keys are years; each record keeps its original date.

.. code-block:: python

   usages_by_year = man_usages.group_by_time()
   usages_by_year.save("corpora_outputs/by_year")
   for year, usages in usages_by_year.items():
       print(year, len(usages))

Printed output:

.. code-block:: text

   1780 76
   1781 114
   1782 281
   1784 9

The counts sum to 480. The ``corpora_outputs/by_year`` directory contains
``1780_usages.jsonl``, ``1781_usages.jsonl``, ``1782_usages.jsonl``, and
``1784_usages.jsonl``. These are usage records, not text summaries of the
counts. The counts alone do not demonstrate a change in meaning.

Outputs saved as JSONL
----------------------

The following are the first three actual usage records written by
``results.save("corpora_outputs")``. There is one JSON object on each line.
Their text, offsets, dates, and other fields come from the program's
exporter without manually rewriting the records.

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 1-3

:download:`Download all exported usages <outputs/corpora_outputs.jsonl>`
or :download:`the sampled usages <outputs/corpora_sample_outputs.jsonl>`.

To regenerate these files, run the code above in order. The local helper
``usecases/tools/export_corpora_tutorial.py`` runs the same search and
native exports from a supplied Parquet path. It saves complete results,
the sample, and year groups in the output directory you specify.
``corpora_examples.py`` remains unchanged. Earlier execution logs are
separate records and are not the usage data shown here.

Summary
-------

We loaded a corpus, searched for a lemma, inspected the returned usages,
and exported actual usage records with the library's JSONL writer. We also
sampled usages and exported year groups while retaining their source
metadata. These operations provide inputs for further study of word usage
and meaning over time.
