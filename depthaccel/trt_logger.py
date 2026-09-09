"""Process-wide singleton TensorRT logger.

TensorRT registers one global logger per process and warns whenever a runtime
or builder is created with a *different* logger object
(``[TRT] [W] The logger passed into createInferRuntime differs from ...``).
All DepthAccel code paths (backend runtimes, engine building, devtools) share
this single instance so the warning never appears.
"""

import tensorrt as trt


_logger = None
_logger_verbose: bool | None = None


def get_trt_logger(verbose: bool = False) -> trt.Logger:
    global _logger, _logger_verbose

    if _logger is None:
        _logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        _logger_verbose = verbose
    elif verbose != _logger_verbose:
        print(
            "[DepthAccel] Requested TensorRT verbose logging, but the process "
            "logger is already active at a different severity; keeping the "
            "existing one.",
            flush=True,
        )
    return _logger
