# tests/test_color_object.py
import pytest

from chromaflow import Color, ColorSpaceError


def test_color_creation() -> None:
    """Tests successful and unsuccessful color creation."""
    c = Color("srgb", (0.1, 0.2, 0.3))
    assert c.space == "srgb"
    assert c.values == (0.1, 0.2, 0.3)

    with pytest.raises(ColorSpaceError):
        Color("unknown-space", (0, 0, 0))


def test_lighten_darken() -> None:
    """Tests the lighten and darken methods."""
    c = Color("oklch", (0.5, 0.1, 120))

    lightened = c.lighten(0.1)
    assert lightened.space == "oklch"
    assert pytest.approx(lightened.values[0]) == 0.6

    darkened = c.darken(0.2)
    assert darkened.space == "oklch"
    assert pytest.approx(darkened.values[0]) == 0.3

    # Test clipping
    white = Color("srgb", (1, 1, 1))
    lightened_white = white.lighten(0.1)

    # After lightening, the color might be out of sRGB gamut.
    # We must clip it back to the gamut to check if it's within [0, 1].
    final_srgb_val = (
        lightened_white.to_gamut("srgb", method="clip").to("srgb").values[0]
    )

    # Assert that the CLIPPED value is now exactly 1.0
    assert final_srgb_val <= 1.0
    assert pytest.approx(final_srgb_val) == 1.0


def test_saturate_desaturate() -> None:
    """Tests saturation methods."""
    c = Color("oklch", (0.5, 0.1, 120))

    saturated = c.saturate(0.05)
    assert saturated.space == "oklch"
    assert pytest.approx(saturated.values[1]) == 0.15

    desaturated = c.desaturate(0.05)
    assert desaturated.space == "oklch"
    assert pytest.approx(desaturated.values[1]) == 0.05

    # Test desaturating to zero
    desaturated_to_gray = c.desaturate(0.1)
    assert pytest.approx(desaturated_to_gray.values[1]) == 0.0


def test_rotate_hue() -> None:
    c = Color("oklch", (0.5, 0.1, 120))

    rotated = c.rotate_hue(90)
    assert pytest.approx(rotated.values[2]) == 210

    # Test wrapping around 360 degrees
    rotated_wrap = c.rotate_hue(300)
    assert pytest.approx(rotated_wrap.values[2]) == (120 + 300) % 360
