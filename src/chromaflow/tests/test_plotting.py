# tests/test_plotting.py
from collections.abc import Iterator

import pytest

from chromaflow import Color, plotting

# Mark tests to be skipped if matplotlib is not installed
plt = pytest.importorskip("matplotlib.pyplot")
try:
    _probe_fig, _probe_ax = plt.subplots()
except Exception:
    pytest.skip(
        "matplotlib.pyplot is not usable in this environment",
        allow_module_level=True,
    )
else:
    plt.close(_probe_fig)


@pytest.fixture(autouse=True)
def close_plots() -> Iterator[None]:
    """Fixture to close all plots after each test to prevent GUI popups."""
    yield
    plt.close("all")


def test_plot_swatch() -> None:
    c1 = Color.from_hex("#FF0000")
    c2 = Color.from_hex("#00FF00")
    fig, ax = plotting.plot_color_swatch([c1, c2], labels=["Red", "Green"])
    assert fig is not None
    assert ax is not None


def test_plot_chromaticity_diagram() -> None:
    fig, ax = plotting.plot_chromaticity_diagram(gamut_footprints=["srgb", "p3-d65"])
    assert fig is not None
    assert ax is not None
