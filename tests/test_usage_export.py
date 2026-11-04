import json

import jsonlines
import numpy as np
import pytest

from languagechange.usages import DWUGUsage, Target, UsageDictionary


@pytest.mark.parametrize('description', [float('nan'), np.float32('nan'), None])
def test_dwug_missing_description_exports_readable_jsonl(tmp_path, description):
    usage = DWUGUsage(
        target=Target('ørret'), date=1929, grouping='1929-1965',
        identifier='usage-1', description=description,
        text='En ørret.', offsets=[3, 8], time=1929,
    )

    UsageDictionary({'ørret': [usage]}).save(str(tmp_path))
    path = tmp_path / 'ørret_usages.jsonl'

    # Reject non-standard JSON constants independently of the reader backend.
    record = json.loads(
        path.read_text(),
        parse_constant=lambda value: pytest.fail(f'Invalid JSON constant: {value}'),
    )
    with jsonlines.open(path) as reader:
        assert list(reader) == [record]
    assert record['description'] == ' '
    assert record['text'] == usage.text()
    assert record['offsets'] == usage.offsets
    assert record['identifier'] == usage.identifier
    assert record['time'] == '1929'
    assert usage.description is description  # Export does not rewrite source data.


@pytest.mark.parametrize('description', ['Provided description', '', ' '])
def test_dwug_description_text_is_preserved(description):
    usage = DWUGUsage(
        target=Target('ørret'), date=1929, grouping='1929-1965',
        identifier='usage-1', description=description,
        text='En ørret.', offsets=[3, 8], time=1929,
    )

    assert usage.to_dict()['description'] == description
