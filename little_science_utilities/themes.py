"""
This module provides color schemes, style dictionaries, and utility functions for
consistent and publication-ready matplotlib plotting. It defines color registries,
color schemes, and a set of style presets for various figure types, as well as context
management for applying styles. Utility functions for exporting figures and creating
fixed-size subplot grids are also included.

Usage:
    - Use ColorRegistry as a registry of predefined colors.
    - Use ColorScheme for consistent color usage across plots.
    - Use Styles context manager to apply a style preset to a matplotlib figure.
    - Use export_for_pub to export figures with publication settings.
    - Use fixed_size_subplots to create subplot grids with fixed dimensions.
"""

import colorsys
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib as mpl
import numpy as np
from matplotlib import RcParams
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

"""
////////////////////////////////////////////////////////////////////////////////////////
// COLORS
////////////////////////////////////////////////////////////////////////////////////////
"""


def set_export_text_type() -> None:
    """
    Set the font type for PDF and PS exports to ensure text is editable in Illustrator.

    :returns: None
    :rtype: None
    """
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]


def export_for_pub(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    """
    Export a figure for publication.

    :param fig: The figure to export.
    :type fig: plt.Figure
    :param path: The path to save the figure to.
    :type path: Path
    :param kwargs: Additional keyword arguments for ``fig.savefig``.
    :type kwargs: Any
    :returns: None
    :rtype: None
    """
    if path.suffix != ".pdf":
        path = path.with_suffix(".pdf")
    fig.savefig(path, transparent=True, **kwargs)


def desaturate(color: "Color", factor: float) -> "Color":
    """
    Desaturate a color by a factor.

    :param color: The color to desaturate.
    :type color: Color
    :param factor: The factor to desaturate by.
    :type factor: float
    :returns: The desaturated color.
    :rtype: Color
    """
    h, l, s = colorsys.rgb_to_hls(color.r, color.g, color.b)  # noqa: E741
    s *= factor
    return Color(*colorsys.hls_to_rgb(h, l, s))


def millimeter_to_inches(mm: float) -> float:
    """
    Convert millimeters to inches.

    :param mm: The value in millimeters.
    :type mm: float
    :returns: The value in inches.
    :rtype: float
    """
    return mm / 25.4


def inches_to_millimeter(inches: float) -> float:
    """
    Convert inches to millimeters.

    :param inches: The value in inches.
    :type inches: float
    :returns: The value in millimeters.
    :rtype: float
    """
    return inches * 25.4


class Color(NamedTuple):
    """
    A color in RGB space.

    :ivar r: Red channel (0-1)
    :vartype r: float
    :ivar g: Green channel (0-1)
    :vartype g: float
    :ivar b: Blue channel (0-1)
    :vartype b: float
    :ivar a: Alpha channel (0-1), default is 1.0
    :vartype a: float
    """

    r: float
    g: float
    b: float
    a: float = 1.0


class ColorRegistry(Enum):
    """
    A registry of colors for different purposes.
    """

    BRIGHT_BLUE = Color(17 / 255, 159 / 255, 255 / 255)
    BRIGHT_RED = Color(255 / 255, 75 / 255, 78 / 255)
    BRIGHT_GREEN = Color(64 / 255, 204 / 255, 139 / 255)
    DESATURATED_BLUE = Color(72 / 255, 136 / 255, 170 / 255)
    DESATURATED_RED = Color(197 / 255, 85 / 255, 94 / 255)
    DESATURATED_GREEN = Color(0 / 255, 158 / 255, 115 / 255)
    DESATURATED_ORANGE = Color(244 / 255, 154 / 255, 95 / 255)
    DESATURATED_PURPLE = Color(101 / 255, 89 / 255, 152 / 255)
    NOTEBOOK_RED = Color(191 / 255, 97 / 255, 106 / 255)
    NOTEBOOK_GREEN = Color(163 / 255, 190 / 255, 140 / 255)
    NOTEBOOK_BLUE = Color(136 / 255, 192 / 255, 208 / 255)
    NOTEBOOK_ORANGE = Color(208 / 255, 135 / 255, 112 / 255)
    NOTEBOOK_YELLOW = Color(235 / 255, 203 / 255, 109 / 255)
    GRAY = Color(128 / 255, 128 / 255, 128 / 255)
    CHARCOAL = Color(51 / 255, 51 / 255, 51 / 255)
    CMO_GREEN = Color(119 / 255, 205 / 255, 162 / 255)
    CMO_PURPLE = Color(64 / 255, 76 / 255, 139 / 255)
    MINT = Color(109 / 255, 209 / 255, 156 / 255)
    BURROW_GREEN = Color(158 / 255, 191 / 255, 164 / 255)
    LIGHT_GRAY = Color(192 / 255, 192 / 255, 192 / 255)
    LIGHTEST_GRAY = Color(228 / 255, 228 / 255, 228 / 255)
    LIGHT_BLUE = Color(102 / 255, 161 / 255, 229 / 255)
    BLUE_BELL = Color(136 / 255, 142 / 255, 201 / 255)
    PINK = Color(255 / 255, 162 / 255, 169 / 255)
    WHITE = Color(1, 1, 1)
    INVISIBLE = Color(0, 0, 0, 0)

    def __call__(self, *args, **kwargs) -> Color:  # noqa: ARG002
        return self.value


class ColorScheme:
    """
    Color scheme for various plot elements.

    This class should not be instantiated.
    """

    # ANNOTATIONS & ADD-ONS
    SCATTER_BORDER: Color = ColorRegistry.CHARCOAL.value
    SECONDARY_LINE: Color = ColorRegistry.CHARCOAL.value
    RECTANGLE_SHADE: Color = Color(*ColorRegistry.LIGHTEST_GRAY.value[:-1], 0.75)
    SIGNIFICANCE: Color = ColorRegistry.CHARCOAL.value
    INVISIBLE: Color = ColorRegistry.INVISIBLE.value
    # DEFAULTS
    DEFAULTS: tuple[Color, ...] = (
        ColorRegistry.DESATURATED_RED.value,
        ColorRegistry.DESATURATED_BLUE.value,
        ColorRegistry.DESATURATED_GREEN.value,
        ColorRegistry.DESATURATED_ORANGE.value,
        ColorRegistry.DESATURATED_PURPLE.value,
    )

    @classmethod
    def get_defaults(cls, idx: int) -> Color:
        """
        Get a default color by index.

        :param idx: Index of the color.
        :type idx: int
        :returns: Color from the defaults.
        :rtype: Color
        """
        return cls.DEFAULTS[idx % len(cls.DEFAULTS)]

    def __new__(cls):
        """
        Prevent instantiation of this class.

        :returns: The class itself.
        :rtype: ColorScheme
        """
        return cls


"""
////////////////////////////////////////////////////////////////////////////////////////
// STYLES
////////////////////////////////////////////////////////////////////////////////////////
"""


PY_GRID: dict = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.linewidth": 1.5,
    "grid.color": (225 / 255, 225 / 255, 225 / 255),
    "axes.linewidth": 2,
    "lines.solid_capstyle": "round",
    "axes.facecolor": "white",
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
    "axes.axisbelow": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.xmargin": 0,
    "axes.ymargin": 0.05,
}


FOV_GRID: dict = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 1.5,
    "grid.alpha": 0.25,
    "axes.linewidth": 2,
    "lines.solid_capstyle": "round",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.xmargin": 0,
    "axes.ymargin": 0.05,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
}


XYZ_FOV: dict = {
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.linewidth": 2,
    "lines.solid_capstyle": "round",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.xmargin": 0,
    "axes.ymargin": 0.05,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
}


D3_GRID: dict = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 1.5,
    "grid.alpha": 0.25,
    "axes.linewidth": 2,
    "lines.solid_capstyle": "round",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
}


BLANK: dict = {
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.linewidth": 0,
    "lines.solid_capstyle": "round",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
}


SINGLE_PANEL_PUB: dict = {
    "figure.figsize": (millimeter_to_inches(55), millimeter_to_inches(51)),
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    ##"savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "ytick.direction": "out",
}


TWO_PANEL_PUB_WIDE: dict = {
    "figure.figsize": (millimeter_to_inches(120), millimeter_to_inches(51)),
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    ##"savefig.bbox": "tight",
    ##"savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "ytick.direction": "out",
}


_PUB: dict = {
    "figure.figsize": (3.3, 2.5),
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans",
        # "Arial",
        "Helvetica",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


PUB2: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


PUB_IMAGE: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 0,
    "ytick.labelsize": 0,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 0,
    "ytick.major.width": 0,
    "ytick.minor.size": 0,
    "ytick.minor.width": 0,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
}


PUB_IMAGE_BOUNDS: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "DejaVu Sans",
        "Arial",
        "Helvetica",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 0,
    "ytick.major.width": 0,
    "ytick.minor.size": 0,
    "ytick.minor.width": 0,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
}


PUB_EMPTY: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 0,
    "ytick.labelsize": 0,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 0,
    "ytick.major.width": 0,
    "ytick.minor.size": 0,
    "ytick.minor.width": 0,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


PUB_VIOLIN: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


PUB_CLUSTER: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
    ],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
}


PUB_BOX_PLOT: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
    "xtick.minor.visible": False,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


PUB_MAP: dict = {
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # "savefig.bbox": "tight",
    # "savefig.pad_inches": 0.01,
    "xtick.direction": "out",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.spines.bottom": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
}


class Styles:
    """
    Context manager for applying matplotlib styles.

    :param style: The style name to apply.
    :type style: str
    """

    __styles__: dict[str, dict | RcParams] = {  # noqa: RUF012
        "py-grid": PY_GRID,
        "fov-grid-light": {**FOV_GRID, "grid.color": "white"},
        "fov-grid-dark": {**FOV_GRID, "grid.color": "black"},
        "xyz-fov": XYZ_FOV,
        "d3-grid": D3_GRID,
        "blank": BLANK,
        "pub": SINGLE_PANEL_PUB,
        "pub-wide": TWO_PANEL_PUB_WIDE,
        "pub2": PUB2,
        "pub-empty": PUB_EMPTY,
        "pub-violin": PUB_VIOLIN,
        "pub-box": PUB_BOX_PLOT,
        "pub-cluster": PUB_CLUSTER,
        "pub-image": PUB_IMAGE,
        "pub-image-bounds": PUB_IMAGE_BOUNDS,
        "pub-map": PUB_MAP,
    }

    def __init__(self, *style: str):
        _style = self.__styles__[style[0]]

        # priority to the first style
        if len(style) > 1:
            for style_ in style:
                _style = {**self.__styles__[style_], **_style}

        self.theme: dict | RcParams | None = _style

    def __enter__(self):
        """
        Enter the style context.

        :returns: The Styles context manager.
        :rtype: Styles
        """
        self._rc_context = mpl.rc_context(self.theme)
        # noinspection PyUnresolvedReferences
        self._rc_context.__enter__()
        # Ensures exported text is editable in Illustrator
        set_export_text_type()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        """
        Exit the style context.

        :param exc_type: Exception type.
        :param exc_val: Exception value.
        :param exc_tb: Exception traceback.
        """
        # noinspection PyUnresolvedReferences
        self._rc_context.__exit__(exc_type, exc_val, exc_tb)
        self.theme = None


def fixed_size_subplots(
    nrows: int,
    ncols: int,
    margin: float = 0.5,
    header: float = 0.5,
    subwidth: float = 2.0,
    subheight: float = 1.75,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create a fixed size subplot grid.

    :param nrows: Number of rows.
    :type nrows: int
    :param ncols: Number of columns.
    :type ncols: int
    :param margin: Margin size.
    :type margin: float
    :param header: Header size.
    :type header: float
    :param subwidth: Width of each subplot.
    :type subwidth: float
    :param subheight: Height of each subplot.
    :type subheight: float
    :returns: Figure and array of Axes.
    :rtype: tuple[plt.Figure, list[plt.Axes]]
    """
    m = margin
    h = header

    a = subheight
    b = subwidth

    width = ncols * (m + b + m)
    height = nrows * (h + a + h)

    axarr = np.empty((nrows, ncols), dtype=object)

    fig = plt.figure(figsize=(width, height))

    for i in range(nrows):
        for j in range(ncols):
            axarr[i, j] = fig.add_axes(
                [
                    (m + j * (2 * m + b)) / width,
                    (height - (i + 1) * (2 * h + a) + h) / height,
                    b / width,
                    a / height,
                ]
            )
    return fig, axarr


def create_custom_colormap(
    colors: list[str], name: str = "custom_colormap"
) -> LinearSegmentedColormap:
    """
    Create a custom colormap using the specified colors.

    :param colors: A list of colors (hex codes or RGB tuples) to include in the colormap.
    :param name: The name of the colormap.
    :return: A LinearSegmentedColormap object.
    """
    return LinearSegmentedColormap.from_list(name, colors)
