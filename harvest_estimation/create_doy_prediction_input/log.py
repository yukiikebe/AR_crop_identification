import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class PipelineLogger:
    def __init__(
        self,
        name,
        log_dir="./logs",
        level=logging.INFO,
        maxBytes=10 * 1024 * 1024,
        backup_count=3,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # Console handler
            sh = logging.StreamHandler()  # terminal output
            sh.setLevel(level)
            sh.setFormatter(formatter)

            # File handler
            fh = RotatingFileHandler(
                self.log_dir / f"{name}.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setLevel(level)  # how much to log to file
            fh.setFormatter(formatter)

            self.logger.addHandler(sh)
            self.logger.addHandler(fh)

        # For manual timing
        self._timers = {}

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)

    # with timing
    def start(self, section_name):
        self.info(f"START {section_name}")
        self._timers[section_name] = time.time()

    def end(self, section_name):
        if section_name not in self._timers:
            self.warning(f"END called for unknown section: {section_name}")
            return

        elapsed = time.time() - self._timers.pop(section_name)
        self.info(f"DONE  {section_name} ({elapsed:.1f}s)")

    def fail(self, section_name):
        self.exception(f"FAIL  {section_name}")
