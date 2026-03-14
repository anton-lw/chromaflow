# src/chromaflow/plotting.py
from __future__ import annotations

import importlib
import io
import pkgutil
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from .color_object import Color
else:
    Axes = Any
    Figure = Any
    Rectangle = Any


def _get_matplotlib() -> tuple[Any, type[Rectangle]]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        patches = importlib.import_module("matplotlib.patches")
    except ImportError as exc:
        raise ImportError(
            "Plotting functionality requires Matplotlib. "
            "Please install it with: pip install chromaflow[plotting]"
        ) from exc

    return plt, cast(type[Rectangle], patches.Rectangle)


def _subplots(plt: Any, **kwargs: Any) -> tuple[Figure, Axes]:
    try:
        result = plt.subplots(**kwargs)
    except Exception as exc:
        raise ImportError("Matplotlib pyplot is unavailable or unusable.") from exc

    if not isinstance(result, tuple) or len(result) != 2:
        raise ImportError("Matplotlib pyplot is unavailable or unusable.")

    fig, ax = result
    return cast(Figure, fig), cast(Axes, ax)


def _load_spectral_locus() -> np.ndarray:
    """Loads the CIE 1931 spectral locus from the package data."""
    # Use pkgutil to reliably access package data files
    data = pkgutil.get_data("chromaflow", "data/cie_1931_2deg_locus.csv")
    if data is None:
        raise FileNotFoundError("Could not find spectral locus data file.")

    # Use io.BytesIO to treat the byte string as a file for numpy
    data_file = io.BytesIO(data)
    # Load columns 1 and 2 (x, y)
    locus = np.loadtxt(data_file, delimiter=",", usecols=(1, 2), comments="#")
    return locus


def plot_color_swatch(
    colors: Color | Sequence[Color],
    labels: list[str] | None = None,
    width: int = 4,
    height: int = 1,
) -> tuple[Figure, Axes]:
    """
    Displays a swatch of one or more colors.

    Args:
        colors: A single Color object or a list of Color objects.
        labels: An optional list of names for each color.
        width: The width of each color patch.
        height: The height of each color patch.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    plt, rectangle = _get_matplotlib()

    if not isinstance(colors, (list, tuple)):
        color_list: list[Color] = [cast(Color, colors)]
    else:
        color_list = list(colors)

    num_colors = len(color_list)
    if labels and len(labels) != num_colors:
        raise ValueError("Number of labels must match number of colors.")

    fig, ax = _subplots(plt, figsize=(width * num_colors, height))

    for i, color in enumerate(color_list):
        srgb_color = color.to("srgb")
        rgb_clipped = np.clip(srgb_color.values, 0, 1)

        ax.add_patch(rectangle((i * width, 0), width, height, color=rgb_clipped))

        if labels:
            try:
                L = color.to("lab-d65").values[0]
                text_color = "white" if L < 50 else "black"
            except Exception:  # Fallback for spaces that can't go to Lab
                text_color = "gray"
            ax.text(
                i * width + width / 2,
                height / 2,
                labels[i],
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    ax.set_xlim(0, num_colors * width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    return fig, ax


def plot_chromaticity_diagram(
    gamut_footprints: list[str] | None = None,
    show_spectral_locus: bool = True,
    show_whitepoints: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plots the CIE 1931 xy chromaticity diagram.

    Args:
        gamut_footprints: A list of RGB color space names to plot as triangles.
        show_spectral_locus: Whether to draw the horseshoe-shaped spectral locus.
        show_whitepoints: Whether to plot the whitepoints of the specified gamuts.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    plt, _ = _get_matplotlib()
    from .core.constants import ILLUMINANT_D65_XY
    from .core.spaces import RGBColorSpace, get_space

    fig, ax = _subplots(plt, figsize=(8, 8.5))

    if gamut_footprints is None:
        gamut_footprints = ["srgb"]

    if show_spectral_locus:
        locus = _load_spectral_locus()
        ax.plot(locus[:, 0], locus[:, 1], color="black", linewidth=1.5, zorder=1)
        ax.plot(
            [locus[-1, 0], locus[0, 0]],
            [locus[-1, 1], locus[0, 1]],
            color="purple",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )

    if gamut_footprints:
        for space_name in gamut_footprints:
            space = get_space(space_name)
            if not isinstance(space, RGBColorSpace):
                continue

            primaries = space.primaries

            # Use primaries to color the edges
            r, g, b = primaries
            ax.plot([r[0], g[0]], [r[1], g[1]], color="red", zorder=2)
            ax.plot([g[0], b[0]], [g[1], b[1]], color="green", zorder=2)
            ax.plot([b[0], r[0]], [b[1], r[1]], color="blue", zorder=2)
            ax.text(r[0], r[1] + 0.02, f"{space.name} R", ha="center")
            ax.text(g[0], g[1] + 0.02, f"{space.name} G", ha="center")
            ax.text(b[0], b[1] - 0.03, f"{space.name} B", ha="center")

    if show_whitepoints:
        # Plot D65 whitepoint
        ax.scatter(
            ILLUMINANT_D65_XY[0],
            ILLUMINANT_D65_XY[1],
            marker="o",
            color="black",
            edgecolor="white",
            s=50,
            zorder=3,
            label="D65",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CIE 1931 Chromaticity Diagram")
    ax.set_aspect("equal")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    return fig, ax
