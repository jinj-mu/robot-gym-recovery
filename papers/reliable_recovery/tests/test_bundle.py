import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundle import current, publish, verify


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.src, self.root = base / "src", base / "bundles"
        self.src.mkdir()
        (self.src / "config.json").write_text('{"version": 1}')
        self.first = publish(self.root, self.src, ["config.json"])

    def test_offline_integrity_and_metadata(self):
        self.assertEqual(current(self.root), self.first)
        self.assertEqual(verify(self.first)["schema"], 1)

    def test_successful_update_retains_old_bundle(self):
        (self.src / "config.json").write_text('{"version": 2}')
        second = publish(self.root, self.src, ["config.json"])
        self.assertEqual(current(self.root), second)
        self.assertEqual(json.loads((self.first / "config.json").read_text())["version"], 1)

    def test_hard_process_exit_at_four_checkpoints(self):
        script = """import os,sys
from bundle import publish
def stop(phase):
    if phase == sys.argv[3]: os._exit(77)
publish(sys.argv[1], sys.argv[2], ['config.json'], checkpoint=stop)
"""
        for phase in ("copied", "verified", "bundle_renamed", "pointer_written"):
            with self.subTest(phase=phase):
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
                r = subprocess.run([sys.executable, "-c", script, str(self.root), str(self.src), phase], env=env)
                self.assertEqual(r.returncode, 77)
                self.assertEqual(current(self.root), self.first)

    def test_corruption_rejected(self):
        (self.first / "config.json").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            current(self.root)

    def test_manifest_change_rejected(self):
        (self.first / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Manifest"):
            current(self.root)

    def test_missing_source_preserves_current(self):
        with self.assertRaises(FileNotFoundError):
            publish(self.root, self.src, ["missing"])
        self.assertEqual(current(self.root), self.first)

    def test_missing_snapshot_file_rejected(self):
        (self.first / "config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            current(self.root)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            publish(self.root, self.src, ["../escape"])
        self.assertEqual(current(self.root), self.first)


if __name__ == "__main__":
    unittest.main()
