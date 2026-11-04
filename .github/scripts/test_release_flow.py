"""Offline tests for release decisions; never contacts GitHub or PyPI."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.old)
        self.addCleanup(self.tmp.cleanup)
        Path('pyproject.toml').write_text('[project]\nname = "langchange"\nversion = "0.2.1"\n')
        Path('dist').mkdir()
        self.files = {'langchange-0.2.1.whl': b'wheel', 'langchange-0.2.1.tar.gz': b'source'}
        for name, data in self.files.items():
            (Path('dist') / name).write_bytes(data)
        self.env = patch.dict(os.environ, {'GITHUB_SHA': 'a' * 40, 'GITHUB_REPOSITORY': 'ChangeIsKey/languagechange', 'GITHUB_OUTPUT': str(Path('output').resolve())}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_versions(self):
        m = load('compare_versions')
        self.assertEqual(m.read_version('pyproject.toml'), '0.2.1')
        Path('bad.toml').write_text('[project]\nversion = "0.2.1+abc"\n')
        with self.assertRaises(ValueError):
            m.read_version('bad.toml')

    def test_version_comparison_detects_change_and_rejects_downgrade(self):
        m = load('compare_versions')
        Path('old.toml').write_text('[project]\nversion = "0.2.0"\n')
        with patch.object(sys, 'argv', ['compare', 'old.toml', 'pyproject.toml']), contextlib.redirect_stdout(io.StringIO()):
            m.main()
        self.assertIn('changed=true', Path('output').read_text())
        Path('old.toml').write_text('[project]\nversion = "0.3.0"\n')
        with patch.object(sys, 'argv', ['compare', 'old.toml', 'pyproject.toml']):
            with self.assertRaises(ValueError):
                m.main()

    def test_evidence_detects_changed_files_and_source(self):
        m = load('release_evidence')
        original = m.evidence()
        (Path('dist') / 'langchange-0.2.1.whl').write_bytes(b'changed')
        self.assertNotEqual(original, m.evidence())
        with patch.dict(os.environ, {'RELEASE_SOURCE_SHA': 'b' * 40}):
            self.assertNotEqual(original['commit'], m.evidence()['commit'])

    def pypi_data(self):
        return {'info': {'name': 'langchange', 'version': '0.2.1'}, 'urls': [
            {'filename': n, 'digests': {'sha256': hashlib.sha256(data).hexdigest()}}
            for n, data in self.files.items()]}

    def test_pypi_matching_and_missing(self):
        m = load('check_pypi_release')
        content = json.dumps(self.pypi_data()).encode()
        with patch.object(sys, 'argv', ['check', '--lookup']), patch.object(m, 'urlopen', return_value=io.BytesIO(content)), contextlib.redirect_stdout(io.StringIO()) as out:
            m.main()
            self.assertEqual(out.getvalue().strip(), 'already-published')
        with patch.object(sys, 'argv', ['check', '--lookup']), patch.object(m, 'urlopen', side_effect=HTTPError('url', 404, 'missing', {}, None)), contextlib.redirect_stdout(io.StringIO()) as out:
            m.main()
            self.assertEqual(out.getvalue().strip(), 'publish')

    def test_pypi_partial_or_offline_stops(self):
        m = load('check_pypi_release')
        data = self.pypi_data()
        data['urls'].pop()
        with patch.object(sys, 'argv', ['check', '--lookup']), patch.object(m, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            with self.assertRaises(ValueError):
                m.main()
        with patch.object(sys, 'argv', ['check', '--lookup']), patch.object(m, 'urlopen', side_effect=URLError('offline')):
            with self.assertRaises(URLError):
                m.main()

    def test_matching_github_release_is_idempotent(self):
        m = load('ensure_github_release')
        ref = {'object': {'type': 'commit', 'sha': 'a' * 40}}
        release = {'tag_name': 'v0.2.1', 'draft': False, 'prerelease': False}
        with patch.object(m, 'api', side_effect=[ref, release]) as api, contextlib.redirect_stdout(io.StringIO()):
            m.main()
        self.assertEqual([c.args[0] for c in api.call_args_list], ['GET', 'GET'])

    def test_missing_github_release_uses_verified_commit(self):
        m = load('ensure_github_release')
        ref = {'object': {'type': 'commit', 'sha': 'a' * 40}}
        with patch.object(m, 'api', side_effect=[None, ref, None, {'tag_name': 'v0.2.1'}]) as api, contextlib.redirect_stdout(io.StringIO()):
            m.main()
        self.assertEqual(api.call_args_list[1].args[2]['sha'], 'a' * 40)
        self.assertEqual(api.call_args_list[3].args[2]['tag_name'], 'v0.2.1')

    def test_github_conflicting_tag_stops(self):
        m = load('ensure_github_release')
        ref = {'object': {'type': 'commit', 'sha': 'b' * 40}}
        with patch.object(m, 'api', return_value=ref) as api:
            with self.assertRaises(ValueError):
                m.main()
        self.assertEqual(api.call_count, 1)

    def test_recovery_run_must_be_successful_main_build(self):
        m = load('validate_recovery_run')
        run = {'event': 'push', 'head_branch': 'main', 'path': '.github/workflows/check-version.yml', 'head_sha': 'a' * 40}
        jobs = {'jobs': [{'name': 'check-version', 'status': 'completed', 'conclusion': 'success'}]}
        Path('run.json').write_text(json.dumps(run)); Path('jobs.json').write_text(json.dumps(jobs))
        with patch.object(sys, 'argv', ['check', 'run.json', 'jobs.json']), patch.dict(os.environ, {'GITHUB_OUTPUT': str(Path('output').resolve())}), contextlib.redirect_stdout(io.StringIO()):
            m.main()
            run['head_branch'] = 'feature'
            Path('run.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                m.main()


if __name__ == '__main__':
    unittest.main()
