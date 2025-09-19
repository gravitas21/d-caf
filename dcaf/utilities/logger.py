import logging
import os
import time
from contextlib import contextmanager


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes on every log record (safer for HPC jobs)."""
    def emit(self, record):
        super().emit(record)
        self.flush()

class TimedLogger:
    """Wrapper that logs labels always, and timings only in DEBUG mode."""
    def __init__(self, logger):
        self.logger = logger

    @contextmanager
    def timing(self, label: str, first_label = True):
        # Always log the label
        if first_label:
            self.logger.info(f"{label}")
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        # Extra timing line only in DEBUG
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"[TIMING] {label} {dt:.6f}")

    # Pass-through methods
    def info(self, *args, **kwargs):    return self.logger.info(*args, **kwargs)
    def warning(self, *args, **kwargs): return self.logger.warning(*args, **kwargs)
    def error(self, *args, **kwargs):   return self.logger.error(*args, **kwargs)
    def debug(self, *args, **kwargs):   return self.logger.debug(*args, **kwargs)
    def critical(self, *args, **kwargs):return self.logger.critical(*args, **kwargs)
    def isEnabledFor(self, level):      return self.logger.isEnabledFor(level)

def setup_logger(output_folder="./dcaf_output/", level=None):
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, "dcaf.log")

    # Reset any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers[:]:
            root.removeHandler(h)

    handler = FlushFileHandler(log_file, mode="w")
    # drop lines that contain "petar: transition from state"
    handler.addFilter(lambda record: "petar: transition from state" not in record.getMessage().lower())

    # Control verbosity via level
    if level is None:
        level_name = os.environ.get("DCAF_LOGLEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[handler]
    )

    base_logger = logging.getLogger("dcaf")
    return TimedLogger(base_logger)
