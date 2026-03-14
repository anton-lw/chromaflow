# src/chromaflow/core/spaces.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..exceptions import ColorSpaceError
from . import constants as const

ColorValueArray = Any


@dataclass(frozen=True)
class ColorSpace:
    name: str


@dataclass(frozen=True)
class RGBColorSpace(ColorSpace):
    primaries: ColorValueArray
    whitepoint: ColorValueArray
    to_linear: Callable[[ColorValueArray], ColorValueArray]
    from_linear: Callable[[ColorValueArray], ColorValueArray]


@dataclass(frozen=True)
class CIEXYZ(ColorSpace):
    whitepoint: ColorValueArray


@dataclass(frozen=True)
class CIELAB(ColorSpace):
    whitepoint: ColorValueArray


_COLOR_SPACE_REGISTRY: dict[str, ColorSpace] = {}


def register_space(space: ColorSpace) -> None:
    name = space.name.lower()
    if name in _COLOR_SPACE_REGISTRY:
        raise ColorSpaceError(f"Color space '{name}' is already registered.")
    _COLOR_SPACE_REGISTRY[name] = space


def get_space(name: str) -> ColorSpace:
    try:
        return _COLOR_SPACE_REGISTRY[name.lower()]
    except KeyError:
        raise ColorSpaceError(f"Color space '{name}' is not registered.") from None


# --- Placeholder Transfer Functions ---
def _srgb_transfer(x: ColorValueArray) -> ColorValueArray:
    raise NotImplementedError("Should be handled by functional API")


def _gamma_decode(x: ColorValueArray, exponent: float) -> ColorValueArray:
    return np.sign(x) * np.power(np.abs(x), exponent)


def _gamma_encode(x: ColorValueArray, exponent: float) -> ColorValueArray:
    return np.sign(x) * np.power(np.abs(x), 1 / exponent)


# sRGB
register_space(
    RGBColorSpace(
        "srgb",
        const.SRGB_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        _srgb_transfer,
        _srgb_transfer,
    )
)
register_space(
    RGBColorSpace(
        "srgb-linear",
        const.SRGB_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        lambda x: x,
        lambda x: x,
    )
)
# CIE Spaces
register_space(CIEXYZ("xyz-d65", const.ILLUMINANT_D65_XY))
register_space(CIELAB("lab-d65", const.ILLUMINANT_D65_XY))
# Oklab/Oklch
register_space(ColorSpace(name="oklab"))
register_space(ColorSpace(name="oklch"))

# Jzazbz
register_space(ColorSpace(name="jzazbz"))

# Other RGB
register_space(
    RGBColorSpace(
        "p3-d65",
        const.P3_D65_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        _srgb_transfer,
        _srgb_transfer,
    )
)
register_space(
    RGBColorSpace(
        "p3-d65-linear",
        const.P3_D65_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        lambda x: x,
        lambda x: x,
    )
)
register_space(
    RGBColorSpace(
        "adobe-rgb",
        const.ADOBE_RGB_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        lambda x: _gamma_decode(x, const.ADOBE_RGB_GAMMA),
        lambda x: _gamma_encode(x, const.ADOBE_RGB_GAMMA),
    )
)
register_space(
    RGBColorSpace(
        "adobe-rgb-linear",
        const.ADOBE_RGB_PRIMARIES_XY,
        const.ILLUMINANT_D65_XY,
        lambda x: x,
        lambda x: x,
    )
)
