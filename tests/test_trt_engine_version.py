"""Tests for the TensorRT engine-version sidecar check."""

from pathlib import Path
import tempfile
import unittest

from depthaccel.tensorrt_backend import check_engine_version, load_tensorrt_model


class EngineVersionTests(unittest.TestCase):
    def _engine_and_sidecar(self, version: str | None):
        tmp = tempfile.mkdtemp()
        engine = Path(tmp) / "da2_vits_fp16_dynamic.trt"
        engine.write_bytes(b"fake-engine")
        if version is not None:
            Path(str(engine) + ".version").write_text(version, encoding="utf-8")
        return engine

    def test_missing_sidecar_is_skipped(self):
        engine = self._engine_and_sidecar(None)
        check_engine_version(engine, "11.3.0.99")  # must not raise

    def test_matching_version_passes(self):
        engine = self._engine_and_sidecar("11.3.0.99")
        check_engine_version(engine, "11.3.0.99")  # must not raise

    def test_version_mismatch_raises(self):
        engine = self._engine_and_sidecar("11.2.1.2")
        with self.assertRaisesRegex(RuntimeError, "11.2.1.2.*11.3.0.99"):
            check_engine_version(engine, "11.3.0.99")

    def test_loader_reports_mismatch_before_deserializing(self):
        try:
            import tensorrt
        except ImportError:
            self.skipTest("tensorrt is not installed")
        # A fake engine cannot deserialize, so only an early check names the version.
        engine = self._engine_and_sidecar("0.0.0")
        with self.assertRaisesRegex(RuntimeError, f"0.0.0.*{tensorrt.__version__}"):
            load_tensorrt_model(engine)


if __name__ == "__main__":
    unittest.main()
