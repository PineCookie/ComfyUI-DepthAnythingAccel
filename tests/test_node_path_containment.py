"""Security regression tests for folder-contained model path resolution.

These lock in the fix that Dropdown model values on DepthAccel nodes must only
ever resolve to files inside ComfyUI's registered ``depthanything`` folder —
never an arbitrary absolute path, a ``..`` traversal, or a file in the repo
artifacts / cwd. See nodes._resolve_existing / _resolve_checkpoint.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
PACKAGE_NAME = "comfyui_depthaccel_path_test_package"


def _load_nodes():
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module.nodes


class FolderContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = _load_nodes()

    def setUp(self):
        # Isolated model folder registered under the shared name.
        import folder_paths

        self._tmp = Path(tempfile.mkdtemp(prefix="depthaccel_path_"))
        self._folder_registered_before = (
            "depthanything" in folder_paths.folder_names_and_paths
        )
        folder_paths.add_model_folder_path("depthanything", str(self._tmp))

    def tearDown(self):
        import folder_paths
        import shutil

        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            paths, _exts = folder_paths.folder_names_and_paths.get(
                "depthanything", ([], set())
            )
            if str(self._tmp) in paths:
                paths.remove(str(self._tmp))
            if not self._folder_registered_before and not paths:
                # Undo the registration this class created so later tests in
                # the same process do not resolve into a stale temp folder.
                folder_paths.folder_names_and_paths.pop("depthanything", None)
        self._tmp = None

    def test_existing_file_in_folder_resolves(self):
        (self._tmp / "da2_vitl_fp16_dynamic.onnx").write_bytes(b"onnx")
        path = self.nodes._resolve_existing(
            "da2_vitl_fp16_dynamic.onnx", (".onnx",)
        )
        self.assertEqual(path, (self._tmp / "da2_vitl_fp16_dynamic.onnx").resolve())

    def test_nested_existing_file_resolves(self):
        nested = self._tmp / "sub"
        nested.mkdir()
        (nested / "model.safetensors").write_bytes(b"sd")
        path = self.nodes._resolve_checkpoint("sub/model.safetensors")
        self.assertEqual(path, (nested / "model.safetensors").resolve())

    def test_traversal_relative_path_is_rejected(self):
        # A real file exists just OUTSIDE the model folder, and the crafted
        # value tries to reach it via "..".
        name = "secret_%s.onnx" % (id(self),)
        outside = self._tmp.parent / name
        outside.write_bytes(b"secret")
        self.addCleanup(outside.unlink, missing_ok=True)
        with self.assertRaises(FileNotFoundError):
            self.nodes._resolve_existing("../" + name, (".onnx",))

    def test_absolute_path_is_rejected(self):
        name = "secret_abs_%s.onnx" % (id(self),)
        outside = self._tmp.parent / name
        outside.write_bytes(b"secret")
        self.addCleanup(outside.unlink, missing_ok=True)
        with self.assertRaises(FileNotFoundError):
            self.nodes._resolve_existing(str(outside), (".onnx",))

    def test_wrong_suffix_is_rejected(self):
        (self._tmp / "foo.pth").write_bytes(b"sd")
        with self.assertRaises(ValueError):
            self.nodes._resolve_existing("foo.pth", (".onnx",))

    def test_unknown_non_downloadable_name_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.nodes._resolve_checkpoint("does_not_exist.safetensors")


if __name__ == "__main__":
    unittest.main()
