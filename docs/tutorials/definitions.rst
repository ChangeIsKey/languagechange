Generating Definitions from Context
===================================

Introduction
------------

A definition generator predicts what a target word means in a sentence.
This tutorial follows the ten English ``plane_nn`` usages selected in
``definition_examples.py`` and uses its T5 XL alternative,
``ltg/flan-t5-definition-en-xl``. We save the actual predictions as JSONL
and turn those definitions into numerical vectors with MPNet. The page
highlights two selected examples: a geometric plane and a woodworking tool.
The complete ten-record output remains available for download.

The definitions shown here were generated in Google Colab on a Tesla T4.
The saved definitions were then encoded with MPNet on a local CPU. These
are separate steps: T5 XL writes definitions, and MPNet encodes their text.
The verification metadata records both environments and model revisions.

.. contents:: In this tutorial
   :local:

Prerequisites
-------------

Install ``languagechange`` and its dependencies. The recorded generation
used library commit ``152a6df796a9900d75df0ed494113feed679f864``; dependency
versions are listed in the downloadable metadata below.

Download and extract `DWUG EN 3.0.0 <https://zenodo.org/records/14028531>`_
so that its ``data`` and ``stats`` folders are inside ``corpora/dwug_en``.
Follow the dataset's reuse conditions. Source sentences are kept in the
full local export and are not bundled in the tutorial downloads.

This example runs generation on a CUDA-capable GPU with at least 13 GiB
of free GPU memory. You can use your own computer, a university server,
or a cloud runtime such as Google Colab. The XL model weights are about
11.4 GB. Encoding the saved definitions can run on a CPU without loading
T5 XL. The first use downloads the respective models.
See the `T5 XL model card
<https://huggingface.co/ltg/flan-t5-definition-en-xl>`_.

The downloadable ``definition_t5_xl_examples.py`` separates these stages.
Save it in a ``variants`` folder and run it from the working directory
containing ``corpora``:

.. code-block:: console

   python variants/definition_t5_xl_examples.py generate
   python variants/definition_t5_xl_examples.py encode

Run ``generate`` in the GPU runtime. Run ``encode`` where the saved
``outputs/definitions_xl/definition_predictions.jsonl`` is available.
The recorded definitions were produced in Colab. Their embeddings were
then computed and saved on a CPU.

Generating context-specific definitions
---------------------------------------

Preparing target usages
~~~~~~~~~~~~~~~~~~~~~~~

Load the dataset and keep the first ten usages in their original order:

.. code-block:: python

   from languagechange.benchmark import DWUG

   dwug = DWUG(path="corpora/dwug_en")
   usages = dwug.get_word_usages("plane_nn")[:10]
   print("Usages:", len(usages))
   print("Target:", usages[0].text()[usages[0].start():usages[0].end()])

Printed output:

.. code-block:: text

   Usages: 10
   Target: plane

``start()`` and ``end()`` are character offsets; the end is excluded.
Keeping the order lets us match each input, prediction, and vector.

Generating definitions with T5 XL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the GPU runtime, download the recorded model revision and initialize
T5 XL without an embedding model. This leaves encoding for the next stage:

.. code-block:: python

   from pathlib import Path
   from huggingface_hub import snapshot_download
   from languagechange.models.representation.definition import T5DefinitionGenerator

   output_dir = Path("outputs/definitions_xl")
   output_dir.mkdir(parents=True, exist_ok=True)
   model_path = snapshot_download(
       "ltg/flan-t5-definition-en-xl",
       revision="f84a7b316a8787a0f1a10f93daf60fc4eb5dda9c",
   )
   definition_generator = T5DefinitionGenerator(
       model_path,
       bsize=1,
       max_length=256,
       filter_target=True,
       sampling=False,
       num_beams=1,
       num_beam_groups=1,
       embedding_model=None,
       cache_dir=str(output_dir / "cache"),
   )
   definition_table = definition_generator.generate_definitions(
       usages, prompt_index=8, return_df=True,
   )
   definition_table = definition_table[["Targets", "Context", "Generated_Definition"]]
   definitions = definition_table["Generated_Definition"].tolist()
   print("Generated definitions:", len(definitions))
   print("First definition:", definitions[0])

Recorded output:

.. code-block:: text

   Generated definitions: 10
   First definition: A flat two-dimensional surface .

``bsize=1`` processes one usage at a time. ``sampling=False`` disables
random sampling. Prompt 8 places the question "What is the definition of
<TRG>?" after the context. ``return_df=True`` returns the actual table;
we select the three export columns explicitly. The T5 constructor does
not take a ``language`` argument. Different dependency versions or hardware
can still affect results, so retain the run metadata.

Saving and checking JSONL records
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write one actual table record per line, with no wrapper around the output:

.. code-block:: python

   definitions_path = output_dir / "definitions.jsonl"
   definition_table.to_json(
       definitions_path, orient="records", lines=True, force_ascii=False,
   )
   predictions_path = output_dir / "definition_predictions.jsonl"
   definition_table[["Targets", "Generated_Definition"]].to_json(
       predictions_path, orient="records", lines=True, force_ascii=False,
   )

The full file keeps the contexts. The second file contains the same target
and prediction values in the same order, without the corpus sentences.
The generator's ``save_definitions()`` method writes TSV, so we use the
DataFrame JSON writer for JSONL.

Read the full export back and compare it with the table:

.. code-block:: python

   import jsonlines

   with jsonlines.open(definitions_path) as reader:
       saved_definitions = list(reader)
   print("Saved records:", len(saved_definitions))
   print("Matches table:", saved_definitions == definition_table.to_dict(orient="records"))

Verified output:

.. code-block:: text

   Saved records: 10
   Matches table: True

Encoding the saved XL definitions
---------------------------------

This step can run in a separate CPU session. Read the saved predictions
rather than generating definitions again. MPNet is the sentence encoder
used by the library to represent definition strings:

.. code-block:: python

   from pathlib import Path
   import jsonlines
   import numpy as np
   import torch
   from sentence_transformers import SentenceTransformer
   from languagechange.models.representation.definition import DefinitionGenerator

   output_dir = Path("outputs/definitions_xl")
   with jsonlines.open(output_dir / "definition_predictions.jsonl") as reader:
       records = list(reader)
   definitions = [row["Generated_Definition"] for row in records]
   torch.set_num_threads(2)
   encoder = DefinitionGenerator(embedding_model=None, cache_dir=None)
   encoder.embedding_model = SentenceTransformer(
       "sentence-transformers/all-mpnet-base-v2",
       revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
       device="cpu",
   )
   embeddings = encoder._encode_definitions(definitions)
   print("Embedding shape:", embeddings.shape)

Verified output:

.. code-block:: text

   Embedding shape: (10, 768)

``_encode_definitions`` is the library's internal encoding method. The
normal combined API calls it when ``return_embeddings=True``. Here we call
it separately so that saved XL predictions can be encoded without reloading
XL. The vectors describe the generated definitions, not the source contexts.
The word ``base`` in the MPNet model name identifies the sentence encoder;
the definition generator used here is T5 XL.

Save and reload the array to verify that no values were changed:

.. code-block:: python

   embeddings_path = output_dir / "definition_embeddings.npy"
   np.save(embeddings_path, embeddings)
   saved_embeddings = np.load(embeddings_path, allow_pickle=False)
   print("Saved shape:", saved_embeddings.shape)
   print("Matches embeddings:", np.array_equal(saved_embeddings, embeddings))
   print("Finite values:", bool(np.isfinite(saved_embeddings).all()))

Verified output:

.. code-block:: text

   Saved shape: (10, 768)
   Matches embeddings: True
   Finite values: True

The saved array contains ten float32 vectors. Repeating the batch encoding
produced the same values. Each definition was also encoded separately to
check that it matched the correct row. The metadata records the small
numerical rounding differences. Row 1 in the array corresponds to JSONL
record 1, and so on.
These checks establish that the vectors match the saved predictions; they
do not establish that the predictions are correct.

Understanding the selected examples
------------------------------------

The ten usages contain different meanings of ``plane``. We select records
1 and 4 to illustrate two clear results from the saved run. Python uses
zero-based positions, so these are rows 0 and 3:

.. code-block:: python

   selected_examples = definition_table.iloc[[0, 3]][["Targets", "Generated_Definition"]]
   print(selected_examples.to_json(orient="records", lines=True, force_ascii=False), end="")

* **Geometry (record 1):** "A flat two-dimensional surface ." describes the
  geometric meaning of ``plane`` in its context.
* **Woodworking (record 4):** the prediction describes a carpenter's hand
  tool for smoothing material. This illustrates a different meaning of
  the same word.

The JSONL records below show the exact generated wording. These examples
were selected for explanation; they are not an accuracy evaluation of
all ten predictions. Some other predictions in the full output are less
accurate. Keep their original strings and check them against their contexts
before using them in an analysis.

Outputs saved as JSONL
----------------------

The two selected XL records (1 and 4) are shown below in their original
order. Each line contains one JSON object. The model strings are kept
exactly as they were returned.

.. literalinclude:: outputs/definitions_outputs.jsonl
   :language: json
   :lines: 1,4

The download retains all ten records in the original order, including
those not displayed here. The embedding array also retains all ten rows;
the selected examples correspond to array rows 0 and 3.

* :download:`All ten XL target and definition records
  <outputs/definitions_outputs.jsonl>`.
* :download:`Corresponding 10 × 768 MPNet vectors
  <outputs/definition_embeddings.npy>`.
* :download:`Model revisions, runtime versions, checksums, and verification results
  <outputs/definitions_run_metadata.json>`.
* :download:`definition_t5_xl_examples.py
  <../../examples/definitions/definition_t5_xl_examples.py>`.

Full local results are saved in ``outputs/definitions_xl`` in the usecases
folder: ``definitions.jsonl``, ``definition_predictions.jsonl``,
``definition_embeddings.npy``, and ``run_metadata.json``. The metadata
belongs to the verified run; the example script does not regenerate that
record of the earlier Colab execution. Printed counts and validation
messages are separate from the JSONL prediction records.

Summary
-------

Use T5 XL to generate definitions, save the returned records as JSONL,
and use MPNet to encode the saved definition strings. Keep the input order,
verify exported files, and check predictions against their contexts before
using their vectors in an analysis.
