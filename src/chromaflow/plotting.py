# src/chromaflow/plotting.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, Tuple
import pkgutil
import io

if TYPE_CHECKING:
    from .color_object import Color

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import Polygon
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

def _check_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting functionality requires Matplotlib. "
            "Please install it with: pip install chromaflow[plotting]"
        )

def _load_spectral_locus() -> np.ndarray:
    """Loads the CIE 1931 spectral locus from the package data."""
    # Use pkgutil to reliably access package data files
    data = pkgutil.get_data("chromaflow", "data/cie_1931_2deg_locus.csv")
    if data is None:
        raise FileNotFoundError("Could not find spectral locus data file.")
    
    # Use io.BytesIO to treat the byte string as a file for numpy
    data_file = io.BytesIO(data)
    # Load columns 1 and 2 (x, y)
    locus = np.loadtxt(data_file, delimiter=',', usecols=(1, 2), comments="#")
    return locus

def plot_color_swatch(
    colors: Union[Color, Sequence[Color]],
    labels: Optional[List[str]] = None,
    width: int = 4,
    height: int = 1
) -> Tuple[Figure, Axes]:
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
    _check_matplotlib()
    
    if not isinstance(colors, (list, tuple)):
        colors = [colors]
    
    num_colors = len(colors)
    if labels and len(labels) != num_colors:
        raise ValueError("Number of labels must match number of colors.")

    fig, ax = plt.subplots(figsize=(width * num_colors, height))

    for i, color in enumerate(colors):
        srgb_color = color.to("srgb")
        rgb_clipped = np.clip(srgb_color.values, 0, 1)
        
        ax.add_patch(plt.Rectangle((i * width, 0), width, height, color=rgb_clipped))
        
        if labels:
            try:
                L = color.to("lab-d65").values[0]
                text_color = 'white' if L < 50 else 'black'
            except Exception: # Fallback for spaces that can't go to Lab
                text_color = 'gray'
            ax.text(
                i * width + width / 2, height / 2, labels[i],
                ha='center', va='center', color=text_color, fontsize=12
            )

    ax.set_xlim(0, num_colors * width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    
    return fig, ax

def plot_chromaticity_diagram(
    gamut_footprints: Optional[List[str]] = ["srgb"],
    show_spectral_locus: bool = True,
    show_whitepoints: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plots the CIE 1931 xy chromaticity diagram.

    Args:
        gamut_footprints: A list of RGB color space names to plot as triangles.
        show_spectral_locus: Whether to draw the horseshoe-shaped spectral locus.
        show_whitepoints: Whether to plot the whitepoints of the specified gamuts.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    _check_matplotlib()
    from .core.spaces import get_space, RGBColorSpace
    from .core.constants import ILLUMINANT_D65_XY

    fig, ax = plt.subplots(figsize=(8, 8.5))

    if show_spectral_locus:
        locus = _load_spectral_locus()
        # Close the loop for the line of purples
        closed_locus = np.vstack([locus, locus[0]])
        ax.plot(locus[:, 0], locus[:, 1], color='black', linewidth=1.5, zorder=1)
        ax.plot(
            [locus[-1, 0], locus[0, 0]],
            [locus[-1, 1], locus[0, 1]],
            color='purple', linestyle='--', linewidth=1, zorder=1
        )

    if gamut_footprints:
        for space_name in gamut_footprints:
            space = get_space(space_name)
            if not isinstance(space, RGBColorSpace):
                continue
            
            primaries = space.primaries
            # Close the triangle
            gamut_poly_coords = np.vstack([primaries, primaries[0]])
            
            # Use primaries to color the edges
            r, g, b = primaries
            ax.plot([r[0], g[0]], [r[1], g[1]], color='red', zorder=2)
            ax.plot([g[0], b[0]], [g[1], b[1]], color='green', zorder=2)
            ax.plot([b[0], r[0]], [b[1], r[1]], color='blue', zorder=2)
            ax.text(r[0], r[1] + 0.02, f"{space.name} R", ha='center')
            ax.text(g[0], g[1] + 0.02, f"{space.name} G", ha='center')
            ax.text(b[0], b[1] - 0.03, f"{space.name} B", ha='center')

    if show_whitepoints:
        # Plot D65 whitepoint
        ax.scatter(
            ILLUMINANT_D65_XY[0], ILLUMINANT_D65_XY[1],
            marker='o', color='black', edgecolor='white', s=50, zorder=3, label='D65'
        )
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CIE 1931 Chromaticity Diagram")
    ax.set_aspect('equal')
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    return fig, ax