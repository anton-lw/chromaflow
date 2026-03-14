# tests/test_conversions.py
from typing import Literal

import numpy as np
import pytest

from chromaflow import Color, config

# A set of test colors covering different parts of the color space
TEST_COLORS = {
    "red": Color.from_hex("#FF0000"),
    "green": Color.from_hex("#00FF00"),
    "blue": Color.from_hex("#0000FF"),
    "white": Color.from_hex("#FFFFFF"),
    "black": Color.from_hex("#000000"),
    "cyan": Color.from_hex("#00FFFF"),
    "orange": Color.from_hex("#D55E00"),
}

# The spaces we want to test round-trips for
ROUND_TRIP_SPACES = ["srgb-linear", "xyz-d65", "lab-d65", "oklab", "oklch", "jzazbz"]

# All available backends
BACKENDS = ["numpy", "jax", "numba"]
BackendName = Literal["numpy", "jax", "numba"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("space", ROUND_TRIP_SPACES)
@pytest.mark.parametrize("color_name", TEST_COLORS.keys())
def test_round_trip_conversions(
    backend: BackendName,
    space: str,
    color_name: str,
) -> None:
    """
    Tests that converting a color to a space and back yields the original color.
    This is a powerful integration test for the entire conversion graph.
    """
    # Skip backends that are not installed
    if backend != "numpy":
        pytest.importorskip(backend)

    original_color = TEST_COLORS[color_name]

    with config.backend(backend):
        # Forward and backward conversion
        intermediate_color = original_color.to(space)
        final_color = intermediate_color.to("srgb")

    # Check if the final values are close to the original values
    original_vals = np.array(original_color.values)
    final_vals = np.array(final_color.values)

    assert np.allclose(original_vals, final_vals, atol=1e-5), (
        f"Round trip failed for {color_name} via {space} with {backend} backend. "
        f"Got {final_vals}, expected {original_vals}"
    )


def test_known_value_conversion() -> None:
    """
    Tests a specific conversion against a known, pre-calculated reference value.
    This ensures our math is correct, not just self-consistent.
    """
    # sRGB (D65) red [1, 0, 0] should convert to CIELAB (D65) with these values.
    # Reference values from http://www.brucelindbloom.com/index.html?ColorCalculator
    srgb_red = Color("srgb", (1, 0, 0))
    lab_red = srgb_red.to("lab-d65")

    expected_lab = (53.2408, 80.0925, 67.2032)

    assert np.allclose(lab_red.values, expected_lab, atol=1e-4)
