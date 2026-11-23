"""No-network onboarding tests with tiny stand-ins for the pinned files."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/npu"))
import prepare_qwen as prep


class PrepareTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.payload = b"Apache-2.0 test fixture"
        self.files = {"LICENSE": hashlib.sha256(self.payload).hexdigest()}
        files = patch.object(prep, "FILES", self.files)
        files.start()
        self.addCleanup(files.stop)

    def test_download_is_pinned_and_reuses_only_complete_verified_files(self):
        with patch.object(prep, "urlopen", return_value=io.BytesIO(self.payload)) as fetch:
            prep.download(self.source)
            self.assertIn(f"/{prep.REVISION}/LICENSE", fetch.call_args.args[0])
            prep.download(self.source)
            self.assertEqual(fetch.call_count, 1)
        (self.source / "LICENSE").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest"):
            prep.download(self.source)
        self.assertEqual((self.source / "LICENSE").read_bytes(), b"changed")

    def test_wrong_download_is_never_published(self):
        with patch.object(prep, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "digest"):
                prep.download(self.source)
        self.assertEqual(list(self.source.iterdir()), [])

    def test_export_refuses_extra_or_substituted_inputs(self):
        (self.source / "LICENSE").write_bytes(self.payload)
        extra = self.source / "custom_generate.py"
        extra.touch()
        with self.assertRaisesRegex(ValueError, "unexpected"):
            prep.verify_source(self.source)
        extra.unlink()
        (self.source / "LICENSE").unlink()
        (self.root / "original").write_bytes(self.payload)
        (self.source / "LICENSE").symlink_to(self.root / "original")
        with self.assertRaisesRegex(ValueError, "regular"):
            prep.verify_source(self.source)

    def test_failure_never_recursively_removes_operator_files(self):
        (self.source / "LICENSE").write_bytes(self.payload)
        output = self.root / "output"
        def failed_tool(*args, **kwargs):
            (output / "keep").write_text("operator file")
            raise subprocess.CalledProcessError(1, "optimum-cli")
        with patch.object(prep.subprocess, "run", side_effect=failed_tool):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.export_model(self.source, output)
        self.assertEqual((output / "keep").read_text(), "operator file")
        self.assertFalse((output / "profile.sha256").exists())

    def test_export_preserves_existing_output_and_pins_complete_new_output(self):
        (self.source / "LICENSE").write_bytes(self.payload)
        output = self.root / "output"

        def fake_export(command, **kwargs):
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(command[command.index("--group-size") + 1], "-1")
            self.assertIn("--sym", command)
            model = Path(command[-1])
            model.mkdir()
            for name in prep.npu_runtime.REQUIRED_FILES:
                (model / name).write_bytes(b"test IR")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(prep.subprocess, "run", side_effect=fake_export) as run, \
             patch.object(prep.subprocess, "check_output", return_value=b"fixture==1\n"):
            prep.export_model(self.source, output)
            with self.assertRaises(FileExistsError):
                prep.export_model(self.source, output)
            self.assertEqual(run.call_count, 1)
        manifest = json.loads((output / "profile.json").read_bytes())
        self.assertEqual(manifest["files"]["SOURCE-LICENSE"], self.files["LICENSE"])
        self.assertEqual(manifest["revision"], prep.REVISION)
        self.assertEqual(prep.npu_runtime.profile_id(manifest), (output / "profile.sha256").read_text().strip())
        self.assertTrue((output / "model/export-freeze.txt").is_file())


if __name__ == "__main__":
    unittest.main()
