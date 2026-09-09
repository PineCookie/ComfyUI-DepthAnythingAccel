"""Run the DA2 ONNX P0 parity, memory, profiling, and shape-stress matrix."""

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import threading
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.onnx_backend import end_onnx_profiling, infer_onnx_raw, load_onnx_model
from depthaccel.preprocessing import prepare_image, restore_depth
from depthaccel.reference import infer_reference_raw, load_reference_model
from devtools.benchmark_reference import load_image, resize_input, synchronize
from devtools.compare_reference_onnx import compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--reference-model", default="depth_anything_v2_vitl_fp32.safetensors")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "p0_validation.json")
    parser.add_argument("--stress-cycles", type=int, default=2)
    parser.add_argument(
        "--skip-native-reference",
        action="store_true",
        help="Skip the expensive native-2K PyTorch case after its boundary is established.",
    )
    parser.add_argument(
        "--skip-native-onnx",
        action="store_true",
        help="Skip native-2K ONNX after its OOM boundary is established.",
    )
    return parser.parse_args()


class GlobalGpuMemorySampler:
    """Sample device-wide memory, including allocations outside PyTorch."""

    def __init__(self, device: torch.device, interval: float = 0.005):
        self.device = device
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.start_used = 0
        self.peak_used = 0

    def _used(self) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        return total - free

    def __enter__(self):
        synchronize(self.device)
        self.start_used = self._used()
        self.peak_used = self.start_used

        def sample():
            while not self.stop_event.wait(self.interval):
                self.peak_used = max(self.peak_used, self._used())

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        synchronize(self.device)
        self.peak_used = max(self.peak_used, self._used())
        self.stop_event.set()
        self.thread.join()

    def result(self) -> dict[str, float]:
        divisor = 1024**2
        return {
            "device_used_at_start_mib": round(self.start_used / divisor, 1),
            "device_peak_used_mib": round(self.peak_used / divisor, 1),
            "peak_delta_mib": round((self.peak_used - self.start_used) / divisor, 1),
        }


def make_cases(images: list[torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    while len(images) < 3:
        images.append(images[len(images) % len(images)])
    return [
        ("square_518", resize_input(images[0], 518, 518)),
        ("landscape_784x1176", resize_input(images[1], 784, 1176)),
        ("portrait_1176x784", resize_input(images[2], 1176, 784)),
        ("odd_source_777x1169", resize_input(images[0], 777, 1169)),
        (
            "batch2_518x770",
            torch.cat(
                [resize_input(images[0], 518, 770), resize_input(images[1], 518, 770)],
                dim=0,
            ),
        ),
        ("native_2k", images[0]),
    ]


def is_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "failed to allocate memory" in message


def profile_summary(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    duration_by_provider = defaultdict(float)
    nodes_by_provider = defaultdict(set)
    duration_by_provider_op = defaultdict(float)
    for event in events:
        args = event.get("args", {})
        provider = args.get("provider")
        if not provider:
            continue
        duration_by_provider[provider] += float(event.get("dur", 0))
        duration_by_provider_op[(provider, args.get("op_name", "unknown"))] += float(
            event.get("dur", 0)
        )
        nodes_by_provider[provider].add(event.get("name", ""))
    return {
        "path": str(path),
        "duration_us_by_provider": dict(sorted(duration_by_provider.items())),
        "event_names_by_provider": {
            provider: len(names) for provider, names in sorted(nodes_by_provider.items())
        },
        "duration_us_by_provider_op": {
            f"{provider}:{operator}": duration
            for (provider, operator), duration in sorted(
                duration_by_provider_op.items(), key=lambda item: item[1], reverse=True
            )
        },
    }


def main() -> None:
    args = parse_args()
    if args.stress_cycles < 1:
        raise ValueError("--stress-cycles must be positive")
    device = torch.device("cuda")
    images = [load_image(path) for path in args.image]
    cases = make_cases(images)
    prepared_cases = []
    for name, image in cases:
        prepared, geometry = prepare_image(image)
        prepared_cases.append((name, prepared, geometry))

    reference = load_reference_model(
        args.reference_model,
        encoder=args.encoder,
        precision="fp32",
        device=device,
        comfyui_root=COMFYUI_ROOT,
    )
    reference_results = {}
    with torch.inference_mode():
        for name, prepared, _ in prepared_cases:
            if name == "native_2k" and args.skip_native_reference:
                reference_results[name] = {"error": "Skipped after confirmed PyTorch OOM boundary."}
                continue
            try:
                reference_results[name] = infer_reference_raw(reference, prepared)
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError as error:
                reference_results[name] = {"error": str(error)}
                torch.cuda.empty_cache()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    profile_prefix = str((REPO_ROOT / "artifacts" / "ort_profile").resolve())
    before_load_used = torch.cuda.mem_get_info(device)
    onnx_model = load_onnx_model(
        args.model,
        device="cuda",
        enable_profiling=True,
        profile_prefix=profile_prefix,
        use_user_compute_stream=True,
    )
    after_load_used = torch.cuda.mem_get_info(device)
    results = []
    with torch.inference_mode():
        for name, prepared, geometry in prepared_cases:
            reference_raw = reference_results[name]
            if name == "native_2k" and args.skip_native_onnx:
                results.append(
                    {
                        "case": name,
                        "status": "skipped_after_confirmed_both_oom",
                    }
                )
                continue
            try:
                with GlobalGpuMemorySampler(device) as memory:
                    started = time.perf_counter()
                    candidate_raw = infer_onnx_raw(onnx_model, prepared)
                    synchronize(device)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                candidate_raw = candidate_raw.cpu()
                result = {
                    "case": name,
                    "status": "onnx_only_reference_oom" if isinstance(reference_raw, dict) else "ok",
                    "prepared_shape": list(prepared.shape),
                    "latency_ms": round(elapsed_ms, 3),
                    "memory": memory.result(),
                }
                if isinstance(reference_raw, dict):
                    result["reference_error"] = reference_raw["error"]
                else:
                    result["raw"] = compare(reference_raw, candidate_raw)
                    result["normalized"] = compare(
                        restore_depth(reference_raw, geometry),
                        restore_depth(candidate_raw, geometry),
                    )
                    if reference_raw.shape[0] > 1:
                        result["raw_per_sample"] = [
                            compare(reference_raw[index : index + 1], candidate_raw[index : index + 1])
                            for index in range(reference_raw.shape[0])
                        ]
                results.append(result)
            except Exception as error:
                if not is_out_of_memory(error):
                    raise
                results.append(
                    {
                        "case": name,
                        "status": "both_oom" if isinstance(reference_raw, dict) else "onnx_oom",
                        "error": str(error),
                    }
                )
                torch.cuda.empty_cache()

        stress = []
        stress_inputs = [case[1] for case in prepared_cases[:4]]
        with GlobalGpuMemorySampler(device) as stress_memory:
            for cycle in range(args.stress_cycles):
                for index, prepared in enumerate(stress_inputs):
                    started = time.perf_counter()
                    output = infer_onnx_raw(onnx_model, prepared)
                    synchronize(device)
                    stress.append(
                        {
                            "cycle": cycle,
                            "shape": list(prepared.shape),
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                            "mean": float(output.mean()),
                        }
                    )

    profile_path = end_onnx_profiling(onnx_model)
    metadata = onnx_model.metadata
    compute_stream = onnx_model.compute_stream
    del onnx_model
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    after_unload_used = torch.cuda.mem_get_info(device)

    def used_mib(snapshot):
        free, total = snapshot
        return round((total - free) / 1024**2, 1)

    report = {
        "model": str(args.model.resolve()),
        "images": [str(path.resolve()) for path in args.image],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_stream": compute_stream,
        "metadata": metadata,
        "matrix": results,
        "stress": {"memory": stress_memory.result(), "runs": stress},
        "session_memory": {
            "before_load_used_mib": used_mib(before_load_used),
            "after_load_used_mib": used_mib(after_load_used),
            "after_unload_used_mib": used_mib(after_unload_used),
        },
        "profile": profile_summary(profile_path),
    }
    import onnxruntime as ort

    report["onnxruntime"] = ort.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
