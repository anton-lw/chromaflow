from __future__ import annotations

import numpy as np

from chromaflow import Color
from chromaflow.core.spaces import _COLOR_SPACE_REGISTRY, RGBColorSpace
from chromaflow.pathfinder import find_conversion_path


def test_all_registered_spaces_are_reachable_from_srgb() -> None:
    sample = Color("srgb", (0.2, 0.4, 0.6))

    for space_name in _COLOR_SPACE_REGISTRY:
        converted = sample.to(space_name)
        round_trip = converted.to("srgb")
        assert np.allclose(round_trip.values, sample.values, atol=1e-5)


def test_all_registered_space_pairs_have_conversion_paths() -> None:
    space_names = tuple(_COLOR_SPACE_REGISTRY)

    for start in space_names:
        for end in space_names:
            assert find_conversion_path(start, end) is not None


def test_all_rgb_spaces_support_gamut_mapping_methods() -> None:
    for space_name, space in _COLOR_SPACE_REGISTRY.items():
        if not isinstance(space, RGBColorSpace):
            continue

        in_gamut = Color(space_name, (0.25, 0.5, 0.75))
        assert in_gamut.in_gamut(space_name) is True

        out_of_gamut = Color(space_name, (1.2, -0.1, 0.5))
        clipped = out_of_gamut.to_gamut(space_name, method="clip")
        mapped = out_of_gamut.to_gamut(space_name, method="oklch-chroma")

        assert clipped.in_gamut(space_name) is True
        assert mapped.in_gamut(space_name) is True
