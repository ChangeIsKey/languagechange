# Definition example

This example uses T5 XL to define the target word in ten English DWUG
contexts. It saves the actual predictions as JSONL and uses MPNet to encode
the saved definitions.

Install `languagechange` from this checkout:

```bash
python -m pip install .
```

Download [DWUG EN 3.0.0](https://zenodo.org/records/14028531) and extract it
so that `corpora/dwug_en` contains the dataset's `data` and `stats` folders.
Follow the dataset's reuse conditions.

Run generation from the working directory containing `corpora`:

```bash
python /path/to/languagechange/examples/definitions/definition_t5_xl_examples.py generate
```

Generation needs a CUDA-capable GPU with at least 13 GiB of free GPU
memory. You can use your own computer, a university server, or a cloud
runtime such as Google Colab. Colab is one option, not a requirement. The
model weights are about 11.4 GB. The example uses a fixed model revision,
batch size 1, and sampling disabled.

Then encode the saved predictions on a CPU:

```bash
python /path/to/languagechange/examples/definitions/definition_t5_xl_examples.py encode
```

The stages can run on different machines. Copy
`outputs/definitions_xl/definition_predictions.jsonl` to the same location
on the CPU machine before running `encode`.

Results are saved in `outputs/definitions_xl/`:

- `definitions.jsonl`: ten targets, contexts, and predictions.
- `definition_predictions.jsonl`: the same targets and predictions, without contexts.
- `definition_embeddings.npy`: ten MPNet vectors in prediction order.
- `cache/`: cached definition results from the generation stage.

The example checks JSONL readback, embedding dimensions, finite values,
and NPY readback. Model predictions still need to be checked against their
contexts. The tutorial shows records 1 and 4 as selected examples, while
its JSONL and embedding downloads retain all ten records.

Model revisions and package versions for the recorded results are in
`docs/tutorials/outputs/definitions_run_metadata.json`. That metadata
describes the recorded Colab generation and CPU embedding checks; the
example script does not overwrite it when rerun.
