# tests/test_gamut.py
from chromaflow import Color

# An sRGB color
IN_GAMUT_SRGB = Color("srgb", (0.5, 0.5, 0.5))
# A color in P3 space that is outside the sRGB gamut (a very vibrant green)
OUT_OF_GAMUT_SRGB = Color("p3-d65", (0, 1, 0))


def test_in_gamut() -> None:
    assert IN_GAMUT_SRGB.in_gamut("srgb") is True
    assert OUT_OF_GAMUT_SRGB.in_gamut("srgb") is False


def test_gamut_clip() -> None:
    clipped = OUT_OF_GAMUT_SRGB.to_gamut("srgb", method="clip")

    # The clipped color must be in gamut
    assert clipped.in_gamut("srgb") is True

    # Check that its values are indeed clipped (will not be the same as original)
    assert clipped.values != OUT_OF_GAMUT_SRGB.values


def test_gamut_oklch_chroma() -> None:
    mapped = OUT_OF_GAMUT_SRGB.to_gamut("srgb", method="oklch-chroma")

    # The mapped color must be in gamut
    assert mapped.in_gamut("srgb") is True

    # Check that its hue is preserved compared to the original
    original_hue = OUT_OF_GAMUT_SRGB.to("oklch").values[2]
    mapped_hue = mapped.to("oklch").values[2]

    # Hue should be very close (allowing for small float precision errors)
    assert abs(original_hue - mapped_hue) < 1e-5
