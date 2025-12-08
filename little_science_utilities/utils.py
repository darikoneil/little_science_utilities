import logging
from enum import Enum, IntEnum
from os import PathLike, getenv
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from tabulate import tabulate

from little_science_utilities.themes import export_for_pub, set_export_text_type


class ListEnum(Enum):
    """
    Enum that allows for list-like behavior.
    """

    @classmethod
    def __getitem__(cls, item: int) -> Any:
        return [member.value for member in cls.__members__.values()][item]

    @classmethod
    def get(cls, item: int) -> Any:
        return cls.__getitem__(item)

    @classmethod
    def __contains__(cls, item: Any) -> bool:
        return item in [member.value for member in cls.__members__.values()]

    @classmethod
    def contains(cls, item: Any) -> bool:
        return cls.__contains__(item)

    @classmethod
    def __index__(cls, item: Any) -> int:
        return [member.value for member in cls.__members__.values()].index(item)

    @classmethod
    def index(cls, item: Any) -> int:
        return cls.__index__(item)

    @classmethod
    def by_value(cls, value: Any, default: Any | None = None) -> Any:
        """
        Get the enum member by its value.
        """
        for member in cls.__members__.values():
            if member.value == value:
                return member
        return default

    @classmethod
    def by_name(cls, name: str, default: Any | None = None) -> Any:
        """
        Get the enum member by its name.
        """
        return cls.__members__.get(name.upper(), default)

    def __eq__(self, other: object):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, Enum):
            return self.value == other.value
        return NotImplemented

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


try:
    _BASE_DIRECTORY = Path(getenv("THESIS_BASE_DIRECTORY"))
except TypeError:
    _BASE_DIRECTORY = Path(R"Z:\My Drive\thesis")


class Directories(dict):
    """
    Class to hold directory paths for different data types.
    """

    def __init__(self, folder: str, base_directory: Path | None = None):
        base_directory = base_directory if base_directory else _BASE_DIRECTORY

        self._folder = folder
        self._base_directory = base_directory

        data = self._base_directory.joinpath("data", folder)
        figures = self._base_directory.joinpath("figures", folder)
        stats = self._base_directory.joinpath("stats", folder)

        for folder_ in (data, figures, stats):
            folder_.mkdir(parents=True, exist_ok=True)

        dict.__init__(
            self,
            data=base_directory.joinpath("data", folder),
            figures=base_directory.joinpath("figures", folder),
            stats=base_directory.joinpath("stats", folder),
        )


def statistics_table_to_file(
    results: pl.DataFrame,
    floatfmt: str = ".3f",
    tablefmt: str = "heavy_grid",
    maxcolwidths: int = 10,
) -> str:
    return tabulate(
        results,
        headers="keys",
        showindex=False,
        floatfmt=floatfmt,
        tablefmt=tablefmt,
        maxcolwidths=maxcolwidths,
    )


class Options(IntEnum):
    SKIP = 0
    SHOW = 1
    SAVE = 2


class ScienceConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter for scientific applications.

    This formatter adds color codes to log messages for better visibility
    in the console. It formats log messages with a timestamp and the message
    content, using a predefined color scheme.
    """

    BLUE: str = "\033[38;2;15;159;255m"
    RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with a custom color scheme.

        :param record: The log record to format.
        :return: The formatted log message as a string.
        """
        log_fmt = f"{self.BLUE}%(asctime)s:\n%(message)s{self.RESET}"
        formatter = logging.Formatter(log_fmt, "%d-%m-%Y %H:%M")
        return formatter.format(record)


class ScienceLogger:
    _LINE_LENGTH = 80
    _LOG_LEVEL = logging.INFO

    def __init__(
        self,
        name: str,
        directory: Path | str | PathLike = _BASE_DIRECTORY,
        figures: Options = Options.SHOW,
        statistics: Options = Options.SHOW,
        integrity: Options = Options.SHOW,
    ):
        self._HEADER = "|" + "=" * (self._LINE_LENGTH - 2) + "|"
        self._SUBHEADER = "|" + "-" * (self._LINE_LENGTH - 2) + "|"
        self._DEMARCATOR = "-" * self._LINE_LENGTH

        self._FIGURES = Options(figures)
        self._STATISTICS = Options(statistics)
        self._INTEGRITY = Options(integrity)

        self.name = name

        self.directory = directory if directory else _BASE_DIRECTORY
        assert self.directory.is_dir()
        assert self.directory.exists()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._LOG_LEVEL)

        if any(
            option >= Options.SHOW
            for option in (self._FIGURES, self._STATISTICS, self._INTEGRITY)
        ):
            self._console_handler = logging.StreamHandler()
            self._console_handler.setLevel(self._LOG_LEVEL)
            self._console_handler.setFormatter(ScienceConsoleFormatter())
            self.logger.addHandler(self._console_handler)

        if any(
            option >= Options.SAVE for option in (self._STATISTICS, self._INTEGRITY)
        ):
            if not self.stats_file.exists():
                with self.stats_file.open("w") as f:
                    f.write("")

            self._file_handler = logging.FileHandler(
                self.stats_file, mode="a", encoding="utf-8"
            )
            self._file_handler.setLevel(self._LOG_LEVEL)
            self._file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(self._file_handler)

    @property
    def data_directory(self) -> Path:
        return self.directory.joinpath(self.name, "data")

    @property
    def figures_directory(self) -> Path:
        return self.directory.joinpath(self.name, "figures")

    @property
    def stats_file(self) -> Path:
        return self.directory.joinpath(self.name, "stats", f"{self.name}_stats.txt")

    def find_data(self, data_name: str) -> Path | None:
        """
        Find a data file in the directory.
        """
        matches = set(self.data_directory.rglob(f"*{data_name}*"))
        if len(matches) > 1:
            msg = (
                f"Multiple files found for {data_name} in "
                f"{self.data_directory}. Using the first match."
            )
            self.logger.warning(msg)
            return None
        if len(matches) == 0:
            msg = f"No files found for {data_name} in {self.data_directory}."
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        return matches.pop()

    def figure(self, fig: plt.Figure, name: str) -> None:
        if self._FIGURES >= Options.SAVE:
            set_export_text_type()
            path = self.figures_directory.joinpath(f"{name}.pdf")
            export_for_pub(fig, path)
            msg = f"Figure saved to {path}"
            self.logger.info(msg)

    def stats(self, message: str | pl.DataFrame | pd.DataFrame) -> None:
        """
        Log a message to the stats stream
        """
        if self._STATISTICS < Options.SHOW:
            return
        if isinstance(message, pl.DataFrame):
            message = message.to_pandas()
        if isinstance(message, pd.DataFrame):
            message = statistics_table_to_file(message)
        msg = f"{self._DEMARCATOR}\n{message}\n{self._DEMARCATOR}"
        self.logger.info(msg)

    def integrity(self, message: str | pl.DataFrame | pd.DataFrame) -> None:
        """
        Log a message to the integrity stream
        """
        if self._INTEGRITY < Options.SHOW:
            return
        msg = f"{self._DEMARCATOR}\n{message}\n{self._DEMARCATOR}"
        self.logger.info(msg)

    def head(self, message: str) -> None:
        """
        Log a header message to the console and stats file.
        """
        header_message = f"{self._HEADER}\n{message}\n{self._HEADER}"
        self.logger.info(header_message)

    def subhead(self, message: str) -> None:
        """
        Log a subheader message to the console and stats file.
        """
        subhead_message = f"{self._SUBHEADER}\n{message}\n{self._SUBHEADER}"
        self.logger.info(subhead_message)
