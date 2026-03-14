from __future__ import annotations

import warnings

import numpy as np
import pytest

from chromaflow import functional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    colour = pytest.importorskip("colour")


RNG = np.random.default_rng(42)
RGB_SAMPLES = RNG.random((64, 3))
XYZ_SAMPLES = colour.RGB_to_XYZ(
    RGB_SAMPLES,
    colour.RGB_COLOURSPACES["sRGB"],
    apply_cctf_decoding=True,
)


def test_random_srgb_xyz_parity_with_colour_science() -> None:
    got = functional.srgb_linear_to_xyz_d65(functional.srgb_to_srgb_linear(RGB_SAMPLES))
    expected = colour.RGB_to_XYZ(
        RGB_SAMPLES,
        colour.RGB_COLOURSPACES["sRGB"],
        apply_cctf_decoding=True,
    )
    # colour-science uses slightly different matrix/reference constants here,
    # so we allow the observed cross-library delta while still catching drift.
    assert np.allclose(got, expected, atol=2e-4, rtol=0.0)


def test_random_display_p3_xyz_parity_with_colour_science() -> None:
    got = functional.p3_d65_linear_to_xyz_d65(
        functional.p3_d65_to_p3_d65_linear(RGB_SAMPLES)
    )
    expected = colour.RGB_to_XYZ(
        RGB_SAMPLES,
        colour.RGB_COLOURSPACES["Display P3"],
        apply_cctf_decoding=True,
    )
    assert np.allclose(got, expected, atol=1e-6, rtol=0.0)


def test_random_adobe_rgb_xyz_parity_with_colour_science() -> None:
    got = functional.adobe_rgb_linear_to_xyz_d65(
        functional.adobe_rgb_to_adobe_rgb_linear(RGB_SAMPLES)
    )
    expected = colour.RGB_to_XYZ(
        RGB_SAMPLES,
        colour.RGB_COLOURSPACES["Adobe RGB (1998)"],
        apply_cctf_decoding=True,
    )
    assert np.allclose(got, expected, atol=1e-5, rtol=0.0)


def test_random_xyz_lab_parity_with_colour_science() -> None:
    got = functional.xyz_d65_to_lab_d65(XYZ_SAMPLES)
    expected = colour.XYZ_to_Lab(XYZ_SAMPLES)
    # Lab parity is dominated by the upstream XYZ/reference-white constants.
    assert np.allclose(got, expected, atol=2e-2, rtol=0.0)


def test_random_xyz_oklab_parity_with_colour_science() -> None:
    got = functional.xyz_d65_to_oklab(XYZ_SAMPLES)
    expected = colour.XYZ_to_Oklab(XYZ_SAMPLES)
    assert np.allclose(got, expected, atol=1e-6, rtol=0.0)


def test_random_xyz_jzazbz_parity_with_colour_science() -> None:
    got = functional.xyz_d65_to_jzazbz(XYZ_SAMPLES)
    expected = colour.XYZ_to_Jzazbz(XYZ_SAMPLES)
    assert np.allclose(got, expected, atol=1e-6, rtol=0.0)
