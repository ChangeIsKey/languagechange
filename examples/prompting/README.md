# Comparing word usages with Groq

This example compares “jam” in strawberry jam, traffic jam and blueberry jam.
It requests DURel scores and WiC decisions from `openai/gpt-oss-120b` through Groq,
then saves the returned fields directly as JSONL.

Install LanguageChange and the provider packages used for the example:

```sh
python -m pip install langchain-groq==1.1.3 groq==0.37.1
```

Configure `GROQ_API_KEY` and `LANGSMITH_API_KEY` privately in the environment.
The current LanguageChange provider setup requests both keys. Do not put keys
in the script or commit them to Git.

From the repository root:

```sh
python examples/prompting/prompting_groq_examples.py --check-setup
python examples/prompting/prompting_groq_examples.py
```

Each run sends four Groq requests. The script uses the library's DURel schema
and strict JSON-schema output for the custom WiC schema. It writes these files
to `outputs/prompting_groq`:

- `durel_predictions.jsonl`: two DURel responses, in pair order.
- `wic_predictions.jsonl`: two WiC responses, in the same order.
- `run_metadata.json`: model, prompts, inputs, package versions and export checks.

The first pair is food/congestion; the second pair is food/food. Execution
succeeded with DURel scores 1 and 4, and WiC labels false and true. The saved
records were read back and matched the responses. New requests can return
different predictions.

The tutorial is in `docs/tutorials/prompting.rst`. Its downloads contain the
actual saved response records and their settings.
