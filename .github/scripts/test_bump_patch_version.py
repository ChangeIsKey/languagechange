"""Exercise version decisions without changing the repository metadata."""

from pathlib import Path
import tempfile
import unittest

from bump_patch_version import prepare


class PatchVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.previous = Path(self.tmp.name) / "previous.toml"
        self.current = Path(self.tmp.name) / "current.toml"

    def write_versions(self, previous, current):
        self.previous.write_text(f'[project]\nversion = "{previous}"\n')
        self.current.write_text(f'[project]\nname = "langchange"\nversion = "{current}" # package\n[other]\nversion = "9.9.9"\n')

    def test_unchanged_increments_patch_and_preserves_other_sections(self):
        self.write_versions("0.2.9", "0.2.9")
        self.assertEqual(prepare(self.previous, self.current), ("0.2.10", True))
        self.assertIn('version = "0.2.10" # package', self.current.read_text())
        self.assertIn('version = "9.9.9"', self.current.read_text())

    def test_manual_minor_change_is_preserved(self):
        self.write_versions("0.2.0", "0.3.0")
        before = self.current.read_text()
        self.assertEqual(prepare(self.previous, self.current), ("0.3.0", False))
        self.assertEqual(self.current.read_text(), before)

    def test_decrease_is_rejected_without_modifying_file(self):
        self.write_versions("0.3.0", "0.2.0")
        before = self.current.read_text()
        with self.assertRaises(ValueError):
            prepare(self.previous, self.current)
        self.assertEqual(self.current.read_text(), before)

    def test_invalid_version_is_rejected(self):
        self.write_versions("0.2.0", "0.3.0.dev1")
        with self.assertRaises(ValueError):
            prepare(self.previous, self.current)


if __name__ == "__main__":
    unittest.main()
