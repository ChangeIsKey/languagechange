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
   :depth: 2

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

Searching for a word
--------------------

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
---------------------

A corpus search can return many examples of a word in context. We can
inspect how many usages were found:

.. code-block:: python

   print(len(man_usages))

Output:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 1

To inspect the first usage, print its sentence and the matched token. The
``start()`` and ``end()`` methods give character offsets into the sentence:

.. code-block:: python

   first_usage = man_usages[0]
   print(first_usage.text())
   print(first_usage.text()[first_usage.start():first_usage.end()])

Output:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 2

We can also sample a smaller number of usages for closer inspection:

.. code-block:: python

   sample = man_usages.sample(10, random_seed=42)
   print(len(sample))
   print(sample[0].text())

Output:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 3

The random seed makes the sample reproducible for the same corpus and
NumPy version. Without a seed, the selected usages can differ between runs.

Grouping usages by time
-----------------------

When the corpus contains time information, usages can be grouped by year:

.. code-block:: python

   usages_by_year = man_usages.group_by_time()
   for year, usages in usages_by_year.items():
       print(year, len(usages))

Output:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json
   :lines: 4

These are the years with matches in this corpus, and the counts sum to 480.
Grouping allows us to inspect usages from different periods. The counts
alone do not demonstrate a change in meaning.

Outputs saved as JSONL
----------------------

The file below contains the selected outputs shown in this tutorial,
formatted as JSONL. Each line is one JSON record with the example name and
its printed output lines. Execution records for ``corpora_examples.py``
are stored separately from these selected tutorial results.

Download :download:`corpora_outputs.jsonl <outputs/corpora_outputs.jsonl>`.
The selected tutorial outputs are shown together below:

.. literalinclude:: outputs/corpora_outputs.jsonl
   :language: json

Summary
-------

In this tutorial, we learned how to load a Parquet corpus, search for a
lemma, inspect the resulting usages, sample examples, and group usages
by time.

These operations provide a starting point for investigating how word
usage and meaning change over time.
