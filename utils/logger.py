import logging
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import ProgressType
from rich.style import Style
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

# Custom theme for Rich
CUSTOM_THEME = Theme(
    {
        "info": Style(color="cyan"),
        "warning": Style(color="yellow"),
        "error": Style(color="red", bold=True),
        "critical": Style(color="red", bold=True, reverse=True),
        "debug": Style(color="green", dim=True),
        "timestamp": Style(color="white", dim=True),
        "logger_name": Style(color="blue"),
    }
)

install_rich_traceback(
    show_locals=False,
    suppress=[],
    width=100,
)


def setup_logging(
    log_file: Optional[str] = None, console: Optional[Console] = None
) -> logging.Logger:
    """Configure logging with Rich formatting."""

    if console is None:
        console = Console(theme=CUSTOM_THEME, force_terminal=True, color_system="auto")

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create logger
    logger = logging.getLogger("skanvision")
    logger.setLevel("INFO")

    # Configure Rich handler for console output
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        tracebacks_suppress=[],
        show_time=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )
    rich_handler.setLevel("INFO")
    logger.addHandler(rich_handler)

    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class CustomLogger:
    """Custom logger with additional formatting and methods."""

    def __init__(self, logger: logging.Logger, console: Console):
        self._logger = logger
        self._console = console

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        self._logger.log(level, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def table(self, data: list, title: str = ""):
        """Display data in a table format."""
        from rich.table import Table

        if not data or not isinstance(data, list):
            return

        table = Table(title=title, show_header=True)

        # Add columns
        headers = data[0].keys() if isinstance(data[0], dict) else None
        if headers:
            for header in headers:
                table.add_column(str(header))

            # Add rows
            for item in data:
                table.add_row(*[str(value) for value in item.values()])

            self._console.print(table)

    def progress(
        self,
        iterable: Iterable[ProgressType] | None = None,
        total=None,
        description="Processing",
    ):
        """Create a progress bar."""
        from rich.progress import track

        if iterable is not None:
            return track(iterable, total=total, description=description)
        return None

    def section(self, title: str):
        """Print a section header."""
        self._console.rule(f"[bold blue]{title}")

    def success(self, msg: str):
        """Log a success message."""
        self._logger.info(f"[bold green] \u2713 [/] {msg}")

    def status(self, msg: str):
        """Create a status spinner."""
        return self._console.status(msg)

    def result(self, success: bool, msg: str):
        """Log an operation result."""
        if success:
            self._logger.info(f"[bold green] \u2713 [/] {msg}")
        else:
            self._logger.error(f"[bold red] X [/] {msg}")


console = Console(
    theme=CUSTOM_THEME, force_terminal=True, color_system="auto", legacy_windows=False
)
base_logger = setup_logging(
    console=console,
)
logger = CustomLogger(base_logger, console)

__all__ = ["logger"]
