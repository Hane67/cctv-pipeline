"""
Logging and profiling utilities for CCTV Video Analytics Pipeline.
"""
import logging
import sys
import time
from functools import wraps
from typing import Callable, Any

# Create logger
logger = logging.getLogger("cctv_pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_log_level(level: int):
    """Set global log level (e.g. logging.DEBUG, logging.INFO)."""
    logger.setLevel(level)


class StageTimer:
    """Context manager to measure and record execution time of pipeline stages."""

    def __init__(self, name: str, timings_dict: dict = None):
        self.name = name
        self.timings_dict = timings_dict
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        if self.timings_dict is not None:
            self.timings_dict[self.name] = self.elapsed_ms


def profile_stage(stage_name: str):
    """Decorator to measure and log execution latency of a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"{stage_name} completed in {ms:.2f} ms")
            return result
        return wrapper
    return decorator
