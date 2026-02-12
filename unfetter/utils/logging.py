"""
Logging and progress bar utilities.

Provides rich console output with progress indicators,
tables, and colored logs for the ablation pipeline.
"""

import logging
import sys
import time
from typing import Optional


def setup_logger(
    name: str = "unfetter",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a configured logger for unfetter.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file to log to.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with rich formatting
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        console.setFormatter(fmt)
        logger.addHandler(console)

    # File handler
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


class ProgressBar:
    """Simple terminal progress bar without rich dependency."""

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 50,
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._render()

    def _render(self):
        """Render progress bar to terminal."""
        if self.total == 0:
            return

        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        rate = self.current / max(elapsed, 0.01)
        remaining = (self.total - self.current) / max(rate, 0.001)

        line = (
            f"\r  {self.desc} [{bar}] "
            f"{pct * 100:.1f}% "
            f"({self.current}/{self.total}) "
            f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
        )
        print(line, end="", flush=True)

        if self.current >= self.total:
            print()  # newline on completion

    def close(self):
        """Finalize the progress bar."""
        if self.current < self.total:
            self.current = self.total
            self._render()
