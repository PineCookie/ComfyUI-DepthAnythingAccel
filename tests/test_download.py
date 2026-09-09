import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from depthaccel.download import (
    MODEL_SOURCES,
    default_model_dir,
    download_model,
    known_model_names,
)


class DownloadTests(unittest.TestCase):
    def test_registry_entries_are_complete(self):
        self.assertGreater(len(known_model_names()), 0)
        for name, source in MODEL_SOURCES.items():
            for key in ("repo_id", "filename", "encoder", "license"):
                self.assertIn(key, source)
                self.assertTrue(source[key])

    def test_unknown_model_raises_without_network(self):
        with self.assertRaises(ValueError):
            download_model("definitely-not-a-model")

    def test_default_model_dir_is_depthanything(self):
        self.assertEqual(default_model_dir().name, "depthanything")

    def test_existing_local_file_is_returned_without_huggingface_hub(self):
        # A present canonical file must load even in an environment without the
        # huggingface_hub package.
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "distill-any-depth-small.safetensors").write_bytes(b"weights")
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
                path = download_model("distill-any-depth-small.safetensors", dest_dir=dest)
            self.assertEqual(path, (dest / "distill-any-depth-small.safetensors").resolve())

    def test_missing_hub_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
                with self.assertRaisesRegex(RuntimeError, "huggingface_hub"):
                    download_model("depth_anything_v2_vits.pth", dest_dir=dest)
            self.assertFalse((dest / "depth_anything_v2_vits.pth").exists())

    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("huggingface_hub"),
        "huggingface_hub is not installed",
    )
    def test_download_stores_file_under_canonical_key_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            fake_cache = Path(tmp) / "fake_hf_cache"
            fake_cache.write_bytes(b"distill weights")
            with mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(fake_cache),
            ) as fake_download:
                path = download_model("distill-any-depth-small.safetensors", dest_dir=dest)

            self.assertEqual(path, (dest / "distill-any-depth-small.safetensors").resolve())
            self.assertEqual(
                (dest / "distill-any-depth-small.safetensors").read_bytes(), b"distill weights"
            )
            # The upstream file lives under a subfolder; the request must use it,
            # while the local copy keeps the canonical flat key name.
            fake_download.assert_called_once_with(
                repo_id="xingyang1/Distill-Any-Depth",
                filename="small/model.safetensors",
            )
            leftovers = [p.name for p in dest.iterdir() if p.name.endswith(".part")]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
