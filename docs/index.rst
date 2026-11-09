.. raw:: html

   <div class="lc-hero">
     <div class="lc-eyebrow">OPEN-SOURCE · PYTHON · LANGUAGE RESEARCH</div>
     <h1>Understand how <span>language changes.</span></h1>
     <p class="lc-lead">A Python toolkit for exploring lexical semantic change across corpora, contexts, and time.</p>
     <div class="lc-actions">
       <a class="lc-button lc-button-primary" href="tutorials/index.html">Explore tutorials <span aria-hidden="true">→</span></a>
       <a class="lc-button lc-button-secondary" href="source/api/languagechange.html">API reference <span aria-hidden="true">↗</span></a>
     </div>
   </div>

.. grid:: 2 2 2 4
   :gutter: 3
   :class-container: lc-features

   .. grid-item-card:: 📚 Corpora & benchmarks
      :link: tutorials/corpora
      :link-type: doc

      Work with historical corpora, SemEval 2020 Task 1, and DWUG datasets.

   .. grid-item-card:: 🧠 Representations
      :link: tutorials/usages
      :link-type: doc

      Explore static and contextualized word representations.

   .. grid-item-card:: 📈 Measure change
      :link: tutorials/timeseries
      :link-type: doc

      Compare meanings across periods with established metrics.

   .. grid-item-card:: 🎨 Visualize results
      :link: tutorials/visualization
      :link-type: doc

      Inspect semantic shifts, embeddings, and change patterns.

Get started
===========

Install LanguageChange from PyPI:

.. code-block:: console

   pip install langchange

New to the toolkit? Start with the :doc:`tutorials/index` or browse the
:doc:`API reference <source/api/languagechange>`.

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Learn & explore

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   source/api/languagechange

About the project
=================

LanguageChange is developed by the `Change is Key! team
<https://github.com/ChangeIsKey/languagechange>`_ with support from
Riksbankens Jubileumsfond (grant M21-0021).

If this library supports your research, please cite:

.. code-block:: bibtex

   @misc{languagechange,
     title = {LanguageChange: A Python library for studying semantic change},
     author = {{Change is Key!}},
     year = {n.d.}
   }
