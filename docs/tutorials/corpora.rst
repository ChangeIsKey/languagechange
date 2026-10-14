Working with Corpora
=====================

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

Before starting, make sure that you have installed ``languagechange``
and have access to a corpus file supported by the package.

Loading a corpus
----------------

LanguageChange supports several corpus formats. For example, ``SprakBankenCorpus``
can load a corpus stored as XML, while ``ParquetCorpus`` can load a corpus stored
in Parquet format.

If you already have a Parquet corpus, load it directly. Replace the path below
with the path to your own file:

.. code-block:: python

   from languagechange.corpora import ParquetCorpus

   corpus = ParquetCorpus("path/to/your-corpus.parquet")

If your corpus is in a supported XML format, you can load it and convert it to
Parquet. For example:

.. code-block:: python

   from languagechange.corpora import SprakBankenCorpus

   xml_corpus = SprakBankenCorpus("path/to/your-corpus.xml")
   corpus = xml_corpus.cast_to_parquet()

Replace ``path/to/your-corpus.xml`` with the path to your own XML corpus file.
The conversion may take some time for large corpora. Consult the corpus
documentation for any format-specific requirements.
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

Exploring word usages
---------------------

A corpus search can return many examples of a word in context. We can
inspect how many usages were found:

.. code-block:: python

   print(len(man_usages))

We can also sample a smaller number of usages for closer inspection:

.. code-block:: python

   sample = man_usages.sample(10)

Grouping usages by time
-----------------------

When the corpus contains time information, usages can be grouped by year:

.. code-block:: python

   usages_by_year = man_usages.group_by_time()

   for year, usages in usages_by_year.items():
       print(year, len(usages))

This allows us to examine how a word appears across different periods.

Summary
-------

In this tutorial, we learned how to load a Parquet corpus, search for a
lemma, inspect the resulting usages, sample examples, and group usages
by time.

These operations provide a starting point for investigating how word
usage and meaning change over time.