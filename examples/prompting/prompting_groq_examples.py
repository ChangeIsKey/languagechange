"""Compare jam usages with Groq and export actual structured responses.

Configure GROQ_API_KEY and LANGSMITH_API_KEY privately before execution.
The current PromptModel provider initializer also requests LangSmith tracing.
"""
import argparse
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='openai/gpt-oss-120b')
parser.add_argument('--output-dir', type=Path, default=Path('outputs/prompting_groq'))
parser.add_argument('--check-setup', action='store_true')
args = parser.parse_args()
missing = [name for name in ['GROQ_API_KEY', 'LANGSMITH_API_KEY'] if not os.environ.get(name)]
if find_spec('langchain_groq') is None:
    missing.append('langchain-groq package')
if args.check_setup:
    print('Model:', args.model)
    print('Missing setup:', ', '.join(missing) if missing else 'none')
    raise SystemExit(1 if missing else 0)
if missing:
    raise SystemExit('Configure the following before running: ' + ', '.join(missing))

import jsonlines
from pydantic import BaseModel, Field
from languagechange.models.representation.prompting import PromptModel
from languagechange.usages import TargetUsage

usages = [
    TargetUsage('Strawberry jam', offsets=[11, 14]),
    TargetUsage('Traffic jam', offsets=[8, 11]),
    TargetUsage('Blueberry jam', offsets=[10, 13]),
]
pairs = [[usages[0], usages[1]], [usages[0], usages[2]]]
user_message = (
    "Please tell me how similar the meaning of the word '{target}' is "
    'in the following example sentences:\n1. {usage_1}\n2. {usage_2}'
)
remote_model = PromptModel(args.model, model_provider='groq')
args.output_dir.mkdir(parents=True, exist_ok=True)
remote_model.set_structure('DURel')
durel_results = []
with jsonlines.open(args.output_dir / 'durel_predictions.jsonl', mode='w') as writer:
    for pair in pairs:
        response = remote_model.get_response(pair, user_prompt_template=user_message)
        record = response.model_dump()
        writer.write(record)
        durel_results.append(record)

class WiCBinary(BaseModel):
    wic_label: bool = Field(description='Whether the word has the same meaning or not.')

remote_model.set_structure(WiCBinary)
# Request schema-constrained JSON instead of relying on a tool call.
remote_model.model = remote_model.llm.with_structured_output(
    WiCBinary, method='json_schema', strict=True,
)
binary_prompt = (
    "Does the word '{target}' have the same meaning in the following sentences?\n"
    '1. {usage_1}\n2. {usage_2}'
)
wic_results = []
with jsonlines.open(args.output_dir / 'wic_predictions.jsonl', mode='w') as writer:
    for pair in pairs:
        response = remote_model.get_response(pair, user_prompt_template=binary_prompt)
        record = response.model_dump()
        writer.write(record)
        wic_results.append(record)
for filename, records in [('durel_predictions.jsonl', durel_results), ('wic_predictions.jsonl', wic_results)]:
    with jsonlines.open(args.output_dir / filename) as reader:
        assert list(reader) == records
metadata = {
    'provider': 'groq', 'model': args.model,
    'durel_output_method': 'function_calling',
    'wic_output_method': 'json_schema', 'wic_strict': True,
    'runtime': {name: version(name) for name in ['langchain-groq', 'groq', 'langchain-core', 'pydantic']},
    'inputs': [{'text': u.text(), 'offsets': u.offsets} for u in usages],
    'pairs': [[0, 1], [0, 2]],
    'durel_prompt': user_message, 'wic_prompt': binary_prompt,
    'system_message': 'You are a lexicographer',
    'readback_matches': True,
}
(args.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('DURel results:', durel_results)
print('WiC results:', wic_results)
print('JSONL readback matches: True')
print('Execution succeeded.')
print('Output folder:', args.output_dir.resolve())
