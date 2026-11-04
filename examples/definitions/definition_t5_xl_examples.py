"""Generate XL definitions on a GPU, or encode saved predictions on a CPU.

Run from the usecases directory containing corpora/dwug_en:
  python variants/definition_t5_xl_examples.py generate
  python variants/definition_t5_xl_examples.py encode
The encode stage reads saved predictions and does not load T5 XL.
"""
import argparse
from pathlib import Path

import jsonlines
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from languagechange.models.representation.definition import DefinitionGenerator, T5DefinitionGenerator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["generate", "encode"])
parser.add_argument("--output-dir", type=Path, default=Path("outputs/definitions_xl"))
args = parser.parse_args()
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
predictions_path = output_dir / "definition_predictions.jsonl"

if args.stage == "generate":
    from huggingface_hub import snapshot_download
    from languagechange.benchmark import DWUG

    if not torch.cuda.is_available():
        raise RuntimeError("Generation needs a CUDA-capable GPU on a workstation, server, or cloud runtime.")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 13 * 1024**3:
        raise RuntimeError("Start with at least 13 GiB of free GPU memory.")
    model_path = snapshot_download(
        "ltg/flan-t5-definition-en-xl",
        revision="f84a7b316a8787a0f1a10f93daf60fc4eb5dda9c",
    )
    dwug = DWUG(path="corpora/dwug_en")
    usages = dwug.get_word_usages("plane_nn")[:10]
    generator = T5DefinitionGenerator(
        model_path, bsize=1, max_length=256, filter_target=True,
        sampling=False, num_beams=1, num_beam_groups=1,
        embedding_model=None, cache_dir=str(output_dir / "cache"),
    )
    table = generator.generate_definitions(usages, prompt_index=8, return_df=True)
    table = table[["Targets", "Context", "Generated_Definition"]]
    table.to_json(output_dir / "definitions.jsonl", orient="records", lines=True, force_ascii=False)
    table[["Targets", "Generated_Definition"]].to_json(
        predictions_path, orient="records", lines=True, force_ascii=False,
    )
    with jsonlines.open(output_dir / "definitions.jsonl") as reader:
        assert list(reader) == table.to_dict(orient="records")
    print("Saved definitions:", len(table))

else:
    with jsonlines.open(predictions_path) as reader:
        records = list(reader)
    if not records or not all(isinstance(row.get("Generated_Definition"), str) for row in records):
        raise ValueError("Expected JSONL records containing Generated_Definition strings.")
    definitions = [row["Generated_Definition"] for row in records]
    torch.set_num_threads(2)
    encoder = DefinitionGenerator(embedding_model=None, cache_dir=None)
    encoder.embedding_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130", device="cpu",
    )
    embeddings = encoder._encode_definitions(definitions)
    assert embeddings.shape == (len(records), 768)
    assert np.isfinite(embeddings).all()
    embeddings_path = output_dir / "definition_embeddings.npy"
    np.save(embeddings_path, embeddings)
    saved = np.load(embeddings_path, allow_pickle=False)
    assert np.array_equal(saved, embeddings)
    print("Embedding shape:", embeddings.shape)
    print("Saved embeddings match:", np.array_equal(saved, embeddings))

print("Output directory:", output_dir)
