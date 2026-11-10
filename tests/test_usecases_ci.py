"""CI-harness tests; these fixtures do not stand in for live model execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_support(name):
    path = ROOT / 'examples/_ci' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_support('run_usecase')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_ci_detection_requires_the_github_actions_flag(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    assert not runner.in_github_actions()
    monkeypatch.setenv('GITHUB_ACTIONS', 'false')
    assert not runner.in_github_actions()
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    assert runner.in_github_actions()


def test_definitions_allow_cpu_in_ci_without_querying_cuda(monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setitem(sys.modules, 'torch', None)
    runner.check_prerequisites('definitions')


def test_missing_api_keys_fail_before_interactive_input(monkeypatch):
    for name in runner.SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match='GROQ_API_KEY, LANGSMITH_API_KEY'):
        runner.check_prerequisites('prompting')


@pytest.mark.parametrize('available,free', [(False, 16), (True, 12)])
def test_definitions_reject_unsuitable_hardware_before_model_loading(monkeypatch, available, free):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    cuda = SimpleNamespace(is_available=lambda: available,
                           mem_get_info=lambda: (free * 1024**3, 16 * 1024**3))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with pytest.raises(ValueError, match='CUDA|13 GiB'):
        runner.check_prerequisites('definitions')


def test_workflow_covers_every_registered_usecase_on_push():
    text = (ROOT / '.github/workflows/usecases.yml').read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    cpu = {row['usecase'] for row in workflow['jobs']['cpu']['strategy']['matrix']['include']}
    assert cpu == set(runner.SCRIPTS)
    assert set(workflow['jobs']) == {'cpu'}
    assert workflow['jobs']['cpu']['runs-on'] == 'ubuntu-24.04'
    assert 'push' in workflow['on']
    assert workflow['on']['push'] in ('', {})
    assert workflow['jobs']['cpu']['strategy']['fail-fast'] == 'false'
    assert 'continue-on-error' not in text


@pytest.mark.parametrize('content', ['', '{"score": NaN}\n', '{"score": Infinity}\n', '[]\n'])
def test_jsonl_rejects_empty_nonstandard_and_nonobject_results(tmp_path, content):
    path = tmp_path / 'results.jsonl'
    path.write_text(content)
    with pytest.raises(ValueError):
        runner.read_jsonl(path)


def test_corpus_is_checksum_verified_and_copied(tmp_path):
    manifest = runner.prepare_corpus(tmp_path)
    source = ROOT / manifest['path']
    copy = tmp_path / 'corpora' / source.name
    assert copy.is_file()
    assert copy.read_bytes() == source.read_bytes()


def test_changed_input_is_not_accepted_silently(tmp_path, monkeypatch):
    (tmp_path / 'examples/_ci').mkdir(parents=True)
    (tmp_path / 'examples/_ci/data_manifest.json').write_text(json.dumps({
        'corpus': {'path': 'input.parquet', 'sha256': 'wrong-checksum'},
    }))
    (tmp_path / 'input.parquet').write_bytes(b'changed input')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='checksum changed'):
        runner.prepare_corpus(tmp_path / 'run')


def test_prompting_checks_schema_without_requiring_fixed_predictions(tmp_path):
    write_jsonl(tmp_path / 'durel_predictions.jsonl', [{'durel': 2}, {'durel': 3}])
    write_jsonl(tmp_path / 'wic_predictions.jsonl', [{'wic_label': True}, {'wic_label': False}])
    result = runner.validate_outputs('prompting', tmp_path, tmp_path, {})
    assert result == {'durel_responses': 2, 'wic_responses': 2}


@pytest.mark.parametrize('bad_label', [1, 0, 'true', None])
def test_prompting_rejects_nonboolean_wic_labels(tmp_path, bad_label):
    write_jsonl(tmp_path / 'durel_predictions.jsonl', [{'durel': 1}, {'durel': 4}])
    write_jsonl(tmp_path / 'wic_predictions.jsonl', [{'wic_label': bad_label}, {'wic_label': True}])
    with pytest.raises(ValueError, match='Boolean labels'):
        runner.validate_outputs('prompting', tmp_path, tmp_path, {})


@pytest.mark.parametrize('bad_score', [0, 5, True, 1.5])
def test_prompting_rejects_invalid_durel_values(tmp_path, bad_score):
    write_jsonl(tmp_path / 'durel_predictions.jsonl', [{'durel': bad_score}, {'durel': 4}])
    write_jsonl(tmp_path / 'wic_predictions.jsonl', [{'wic_label': False}, {'wic_label': True}])
    with pytest.raises(ValueError, match='DURel'):
        runner.validate_outputs('prompting', tmp_path, tmp_path, {})


def test_definition_commands_generate_before_encoding_and_keep_xl(tmp_path):
    commands = runner.commands('definitions', tmp_path)
    assert [command[2] for command in commands] == ['generate', 'encode']
    assert all(command[1].endswith('definition_t5_xl_examples.py') for command in commands)
    assert '--download-dataset' in commands[0]
    assert not any('--reuse-run' in command for command in commands)


def test_arrays_must_be_finite_and_have_the_required_shape(tmp_path):
    path = tmp_path / 'embeddings.npy'
    np.save(path, np.full((10, 768), np.nan))
    with pytest.raises(ValueError, match='Invalid embedding'):
        runner.check_array(path, (10, 768))
    np.save(path, np.ones((9, 768)))
    with pytest.raises(ValueError, match='Invalid embedding'):
        runner.check_array(path, (10, 768))


def test_child_failure_is_reported_and_returns_failure(tmp_path, monkeypatch):
    script = tmp_path / 'fails.py'
    script.write_text('raise SystemExit(7)\n')
    monkeypatch.setattr(runner, 'commands', lambda topic, output: [[sys.executable, str(script)]])
    destination = tmp_path / 'fresh-run'
    assert runner.main(['corpora', '--output-dir', str(destination)]) == 1
    report = json.loads((destination / 'run.json').read_text())
    assert report['status'] == 'failed'
    assert report['error_type'] == 'CalledProcessError'
    assert 'checks' not in report


def test_zero_exit_without_required_outputs_still_fails(tmp_path, monkeypatch):
    script = tmp_path / 'does_nothing.py'
    script.write_text('pass\n')
    monkeypatch.setattr(runner, 'commands', lambda topic, output: [[sys.executable, str(script)]])
    monkeypatch.setenv('GROQ_API_KEY', 'test-groq-placeholder')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'test-langsmith-placeholder')
    destination = tmp_path / 'fresh-run'
    assert runner.main(['prompting', '--output-dir', str(destination)]) == 1
    report = json.loads((destination / 'run.json').read_text())
    assert report['status'] == 'failed'
    assert report['error_type'] == 'FileNotFoundError'


def test_old_output_directory_cannot_be_replayed_as_a_new_run(tmp_path):
    (tmp_path / 'run.json').write_text('{"status": "passed"}')
    with pytest.raises(ValueError, match='new or empty'):
        runner.main(['corpora', '--output-dir', str(tmp_path)])
    assert json.loads((tmp_path / 'run.json').read_text()) == {'status': 'passed'}


def test_known_secrets_are_redacted_from_saved_logs(tmp_path, monkeypatch, capsys):
    token = 'example-test-credential-not-a-real-key'
    monkeypatch.setenv('GROQ_API_KEY', token)
    script = tmp_path / 'print_environment.py'
    script.write_text('import os\nprint(os.environ["GROQ_API_KEY"])\n')
    logfile = tmp_path / 'stage.log'
    runner.run_command([sys.executable, str(script)], tmp_path, logfile)
    assert token not in logfile.read_text()
    assert '[REDACTED]' in logfile.read_text()
    assert token not in capsys.readouterr().out


def test_truncated_graph_is_rejected(tmp_path):
    path = tmp_path / 'graph.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(OSError):
        runner.check_png(path)


def test_ci_sample_keeps_all_four_years_and_matches_script_selection(tmp_path):
    runner.prepare_corpus(tmp_path)
    usages = runner.source_usages(tmp_path)
    sample = runner.ci_sample(usages)
    assert len(sample) == 10
    assert [len(group) for _, group in sorted(sample.group_by_time().items())] == [3, 3, 2, 2]
    assert [u.text() for u in sample] == [u.text() for u in runner.ci_sample(usages)]


def test_ci_child_process_has_no_visible_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    script = tmp_path / 'device.py'
    script.write_text('import os\nassert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n')
    runner.run_command([sys.executable, str(script)], tmp_path, tmp_path / 'device.log')
