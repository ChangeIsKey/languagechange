"""Check recovery rejects source commits unrelated to the original run."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from resolve_recovery_source import resolve


class RecoverySourceTests(unittest.TestCase):
    def test_recorded_original_commit_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            sha = 'a' * 40
            manifest.write_text(json.dumps({'commit': sha}))
            self.assertEqual(resolve(sha, manifest), sha)

    def test_automatic_child_must_change_only_pyproject(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            before, child = 'a' * 40, 'b' * 40
            manifest.write_text(json.dumps({'commit': child}))
            with patch('subprocess.check_output', side_effect=[f'{child} {before}\n', 'pyproject.toml\n']):
                self.assertEqual(resolve(before, manifest), child)
            with patch('subprocess.check_output', side_effect=[f'{child} {before}\n', 'README.md\n']):
                with self.assertRaises(ValueError):
                    resolve(before, manifest)
            with patch('subprocess.check_output', return_value=f'{child} {"c" * 40}\n'):
                with self.assertRaises(ValueError):
                    resolve(before, manifest)


if __name__ == '__main__':
    unittest.main()
