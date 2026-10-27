Comparing Usages with Prompting
===============================

Introduction
------------

This tutorial follows ``prompting_groq_examples.py``
to compare the meaning of ``jam`` in two pairs of contexts. We request a
DURel relatedness score and a word-in-context (WiC) decision, then save the
actual structured responses as JSONL.

The example uses ``openai/gpt-oss-120b`` through Groq. The saved responses
come from the completed execution with this model.

.. contents:: In this tutorial
   :local:

Prerequisites
-------------

Install ``languagechange`` and the Groq integration. The execution used:

.. code-block:: console

   python -m pip install langchain-groq==1.1.3 groq==0.37.1

Groq runs the model on its servers, so no local model download or GPU is
needed. Create an API key in the `Groq console
<https://console.groq.com/keys>`_. The model is listed under Groq's
`free-plan limits <https://console.groq.com/docs/rate-limits>`_; availability
and account limits can change.

The current ``PromptModel`` provider initializer also enables LangSmith
tracing and requests ``LANGSMITH_API_KEY``. This is a requirement of this
library setup, separate from Groq authentication. Follow the
`LangSmith account and API-key instructions
<https://docs.langchain.com/langsmith/create-account-api-key>`_.

Run the Python blocks in order in one session. The downloadable
``prompting_groq_examples.py`` contains the complete example. When using
that script, configure both keys in its environment first. Keys entered
inside a Python session remain in that session and its child processes;
a separate terminal process does not inherit them from an already running
Python session.

Setting up the example
----------------------

Configuring keys privately
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use hidden input rather than putting credentials in the script:

.. code-block:: python

   import os
   from getpass import getpass

   if not os.environ.get("GROQ_API_KEY"):
       os.environ["GROQ_API_KEY"] = getpass("Groq API key: ").strip()
   if not os.environ.get("LANGSMITH_API_KEY"):
       os.environ["LANGSMITH_API_KEY"] = getpass("LangSmith API key: ").strip()

A nonempty value does not establish that a key is valid. A Groq
``401 Invalid API Key`` response means authentication failed. Re-enter the
correct key and recreate the model client, which otherwise retains the
previous key. Keep credentials out of output files and version control.

Preparing the usage pairs
~~~~~~~~~~~~~~~~~~~~~~~~~

``TargetUsage`` holds a context and the target's character offsets.
The end offset is excluded:

.. code-block:: python

   from languagechange.usages import TargetUsage

   target_usages = [
       TargetUsage("Strawberry jam", offsets=[11, 14]),
       TargetUsage("Traffic jam", offsets=[8, 11]),
       TargetUsage("Blueberry jam", offsets=[10, 13]),
   ]
   for usage in target_usages:
       print(usage.text()[usage.start():usage.end()])

Printed output:

.. code-block:: text

   jam
   jam
   jam

Pair 1 compares food with congestion. Pair 2 compares two food usages.
Both result files retain this order.

Initializing the provider model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select Groq and the model used for this execution:

.. code-block:: python

   from languagechange.models.representation.prompting import PromptModel

   remote_model = PromptModel(
       "openai/gpt-oss-120b",
       model_provider="groq",
   )
   remote_model.set_structure("DURel")

``set_structure("DURel")`` requests the library's schema containing an
integer ``durel`` field. Its scale is 1 for unrelated meanings, 2 for
distantly related meanings, 3 for closely related meanings, and 4 for
identical meanings. The schema description supplies this scale to the model.

Comparing meanings
------------------

Requesting DURel scores
~~~~~~~~~~~~~~~~~~~~~~~

The placeholders in the prompt are filled from the two target usages:

.. code-block:: python

   user_message = (
       "Please tell me how similar the meaning of the word '{target}' is "
       "in the following example sentences:\n1. {usage_1}\n2. {usage_2}"
   )
   r1 = remote_model.get_response(
       [target_usages[0], target_usages[1]],
       user_prompt_template=user_message,
   )
   r2 = remote_model.get_response(
       [target_usages[0], target_usages[2]],
       user_prompt_template=user_message,
   )
   print("Food / congestion:", r1.durel)
   print("Food / food:", r2.durel)

Output from the completed execution:

.. code-block:: text

   Food / congestion: 1
   Food / food: 4

These responses fit the example distinction: the first pair uses unrelated
meanings and the second shares the food meaning. They are two illustrative
predictions, not a measured accuracy score. Responses may differ between
requests or when a provider updates its model.

Requesting a WiC decision
~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom schema for a boolean decision about the same meaning:

.. code-block:: python

   from pydantic import BaseModel, Field

   class WiCBinary(BaseModel):
       wic_label: bool = Field(description="Whether the word has the same meaning or not.")

   remote_model.set_structure(WiCBinary)
   remote_model.model = remote_model.llm.with_structured_output(
       WiCBinary,
       method="json_schema",
       strict=True,
   )

The current library's ``set_structure()`` uses the integration's default
function-calling mode. During execution, the model sometimes answered in
prose instead of calling the required tool, producing ``tool_use_failed``.
The explicit binding above selects Groq's strict JSON-schema mode, which
worked for this model and schema. See `Groq structured outputs
<https://console.groq.com/docs/structured-outputs>`_. This selects the
response format using the provider integration; no library implementation
change is required for the example.

Now request the boolean decisions:

.. code-block:: python

   binary_prompt = (
       "Does the word '{target}' have the same meaning in the following sentences?\n"
       "1. {usage_1}\n2. {usage_2}"
   )
   wic1 = remote_model.get_response(
       [target_usages[0], target_usages[1]],
       user_prompt_template=binary_prompt,
   )
   wic2 = remote_model.get_response(
       [target_usages[0], target_usages[2]],
       user_prompt_template=binary_prompt,
   )
   print("Food / congestion:", wic1.wic_label)
   print("Food / food:", wic2.wic_label)

Output from the completed execution:

.. code-block:: text

   Food / congestion: False
   Food / food: True

``False`` indicates different meanings; ``True`` indicates the same
meaning. The schema controls the structure and types of the answer.
Inspecting the contexts is still necessary to assess whether the prediction
makes sense.

Saving and checking responses
-----------------------------

Saving actual responses as JSONL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``model_dump()`` returns each response's actual schema fields. Save these
objects directly, one per line:

.. code-block:: python

   from pathlib import Path
   import jsonlines

   output_dir = Path("outputs/prompting_groq")
   output_dir.mkdir(parents=True, exist_ok=True)
   durel_records = [r1.model_dump(), r2.model_dump()]
   wic_records = [wic1.model_dump(), wic2.model_dump()]
   durel_path = output_dir / "durel_predictions.jsonl"
   wic_path = output_dir / "wic_predictions.jsonl"
   with jsonlines.open(durel_path, mode="w") as writer:
       writer.write_all(durel_records)
   with jsonlines.open(wic_path, mode="w") as writer:
       writer.write_all(wic_records)

The files contain the response fields ``durel`` and ``wic_label``.
Printed counts, prompts, and validation messages are kept separate.
Python's ``True`` and ``False`` are written as JSON ``true`` and ``false``.

Checking saved records
~~~~~~~~~~~~~~~~~~~~~~

Read both files back and compare them with the response objects:

.. code-block:: python

   with jsonlines.open(durel_path) as reader:
       saved_durel = list(reader)
   with jsonlines.open(wic_path) as reader:
       saved_wic = list(reader)
   print("Saved DURel records:", len(saved_durel))
   print("DURel matches responses:", saved_durel == durel_records)
   print("Saved WiC records:", len(saved_wic))
   print("WiC matches responses:", saved_wic == wic_records)

Verified output:

.. code-block:: text

   Saved DURel records: 2
   DURel matches responses: True
   Saved WiC records: 2
   WiC matches responses: True

Execution succeeded. These checks confirm that the JSONL files can be
parsed and preserve the actual response values.

Running the downloadable example
---------------------------------

Save :download:`prompting_groq_examples.py
<../../examples/prompting/prompting_groq_examples.py>` in a ``variants``
folder inside your usecases directory. With both keys configured in the
current Python session, run:

.. code-block:: python

   import runpy

   # This sends four new Groq requests and saves their actual responses.
   runpy.run_path("variants/prompting_groq_examples.py", run_name="__main__")

Alternatively, from a terminal that already has both keys configured:

.. code-block:: console

   python variants/prompting_groq_examples.py

The script writes both prediction files and ``run_metadata.json`` with
the model, prompts, pair order, output methods, package versions, and
read-back check. A new run can return different predictions from the
recorded examples. The metadata download below describes the completed
execution and its saved-file verification; it contains no API credentials.

Outputs saved as JSONL
----------------------

Actual DURel responses, first pair followed by second pair:

.. literalinclude:: outputs/prompting_outputs.jsonl
   :language: json

Actual WiC responses in the same order:

.. literalinclude:: outputs/prompting_wic_outputs.jsonl
   :language: json

* :download:`Two DURel response records <outputs/prompting_outputs.jsonl>`.
* :download:`Two WiC response records <outputs/prompting_wic_outputs.jsonl>`.
* :download:`Model, prompts, response methods, and verification metadata
  <outputs/prompting_run_metadata.json>`.
* :download:`prompting_groq_examples.py
  <../../examples/prompting/prompting_groq_examples.py>`.

Full response files are saved in ``outputs/prompting_groq`` in your working
folder.

Summary
-------

Prepare usage pairs, select a provider model, and request relatedness scores
or boolean decisions using schemas. For this WiC workflow, select strict
JSON-schema output explicitly. Save the actual response objects as JSONL,
verify them by reading the files back, and interpret each result against
its contexts.
