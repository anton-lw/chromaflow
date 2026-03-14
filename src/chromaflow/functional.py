# src/chromaflow/functional.py
from importlib import import_module
from typing import Any

import numpy as np

from . import config
from .core.constants import (
    ADOBE_RGB_GAMMA,
    ADOBE_RGB_TO_XYZ_MATRIX,
    ILLUMINANT_D65_XYZ,
    P3_TO_XYZ_MATRIX,
    SRGB_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_RGB_MATRIX,
    XYZ_TO_P3_MATRIX,
    XYZ_TO_SRGB_MATRIX,
)
from .exceptions import BackendConfigurationError

_DISPATCH = {
    "numpy": "chromaflow.backends.numpy_backend",
    "jax": "chromaflow.backends.jax_backend",
    "numba": "chromaflow.backends.numba_backend",
}
_MODULE_CACHE: dict[str, Any] = {}


def _sign_preserving_power(values: np.ndarray, exponent: float) -> np.ndarray:
    result = np.sign(values) * np.power(np.abs(values), exponent)
    return np.asarray(result)


def _get_backend_module(name: str) -> Any:
    try:
        module_name = _DISPATCH[name]
    except KeyError:
        raise BackendConfigurationError(
            f"Backend '{name}' not found or installed."
        ) from None

    if name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[name] = import_module(module_name)
        except ImportError as exc:
            raise BackendConfigurationError(
                f"Backend '{name}' is not installed."
            ) from exc

    return _MODULE_CACHE[name]


def _get_backend_func(name: str) -> Any:
    """Dynamically get a function from the currently configured backend."""
    backend_name = config.get_backend()
    backend_module = _get_backend_module(backend_name)
    func = getattr(backend_module, name, None)
    if func is None:
        # Fallback to numpy if the function is not implemented in the selected backend
        numpy_backend = _get_backend_module("numpy")
        func = getattr(numpy_backend, name, None)
        if func is None:
            raise NotImplementedError(
                f"Function '{name}' not implemented for backend "
                f"'{backend_name}' or numpy fallback."
            )
    return func


def srgb_to_srgb_linear(srgb: np.ndarray) -> np.ndarray:
    """Converts non-linear sRGB to linear sRGB."""
    func = _get_backend_func("srgb_eotf")
    return np.asarray(func(srgb))


def srgb_linear_to_srgb(srgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear sRGB to non-linear sRGB."""
    func = _get_backend_func("srgb_oetf")
    return np.asarray(func(srgb_linear))


def p3_d65_to_p3_d65_linear(p3: np.ndarray) -> np.ndarray:
    """Converts non-linear Display P3 to linear Display P3."""
    # Display P3 uses the same transfer function as sRGB.
    return srgb_to_srgb_linear(p3)


def p3_d65_linear_to_p3_d65(p3_linear: np.ndarray) -> np.ndarray:
    """Converts linear Display P3 to non-linear Display P3."""
    # Display P3 uses the same transfer function as sRGB.
    return srgb_linear_to_srgb(p3_linear)


def adobe_rgb_to_adobe_rgb_linear(adobe_rgb: np.ndarray) -> np.ndarray:
    """Converts non-linear Adobe RGB (1998) to linear Adobe RGB."""
    return _sign_preserving_power(adobe_rgb, ADOBE_RGB_GAMMA)


def adobe_rgb_linear_to_adobe_rgb(adobe_rgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear Adobe RGB (1998) to non-linear Adobe RGB."""
    return _sign_preserving_power(adobe_rgb_linear, 1 / ADOBE_RGB_GAMMA)


def p3_d65_linear_to_xyz_d65(p3_linear: np.ndarray) -> np.ndarray:
    """Converts linear Display P3 to CIE XYZ (D65)."""
    func = _get_backend_func("rgb_to_xyz")
    return np.asarray(func(p3_linear, P3_TO_XYZ_MATRIX))


def adobe_rgb_linear_to_xyz_d65(adobe_rgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear Adobe RGB (1998) to CIE XYZ (D65)."""
    func = _get_backend_func("rgb_to_xyz")
    return np.asarray(func(adobe_rgb_linear, ADOBE_RGB_TO_XYZ_MATRIX))


def xyz_d65_to_p3_d65_linear(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to linear Display P3."""
    func = _get_backend_func("xyz_to_rgb")
    return np.asarray(func(xyz, XYZ_TO_P3_MATRIX))


def xyz_d65_to_adobe_rgb_linear(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to linear Adobe RGB (1998)."""
    func = _get_backend_func("xyz_to_rgb")
    return np.asarray(func(xyz, XYZ_TO_ADOBE_RGB_MATRIX))


def srgb_linear_to_xyz_d65(srgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear sRGB to CIE XYZ (D65)."""
    func = _get_backend_func("rgb_to_xyz")
    return np.asarray(func(srgb_linear, SRGB_TO_XYZ_MATRIX))


def xyz_d65_to_srgb_linear(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to linear sRGB."""
    func = _get_backend_func("xyz_to_rgb")
    return np.asarray(func(xyz, XYZ_TO_SRGB_MATRIX))


def xyz_d65_to_lab_d65(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ to CIELAB, assuming a D65 illuminant."""
    func = _get_backend_func("xyz_to_lab")
    return np.asarray(func(xyz, ILLUMINANT_D65_XYZ))


def lab_d65_to_xyz_d65(lab: np.ndarray) -> np.ndarray:
    """Converts CIELAB to CIE XYZ, assuming a D65 illuminant."""
    func = _get_backend_func("lab_to_xyz")
    return np.asarray(func(lab, ILLUMINANT_D65_XYZ))


def xyz_d65_to_oklab(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to Oklab."""
    func = _get_backend_func("xyz_to_oklab")
    return np.asarray(func(xyz))


def oklab_to_xyz_d65(oklab: np.ndarray) -> np.ndarray:
    """Converts Oklab to CIE XYZ (D65)."""
    func = _get_backend_func("oklab_to_xyz")
    return np.asarray(func(oklab))


def oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    """Converts Oklab to Oklch (cylindrical coordinates)."""
    func = _get_backend_func("oklab_to_oklch")
    return np.asarray(func(oklab))


def oklch_to_oklab(oklch: np.ndarray) -> np.ndarray:
    """Converts Oklch to Oklab (cartesian coordinates)."""
    func = _get_backend_func("oklch_to_oklab")
    return np.asarray(func(oklch))


def xyz_d65_to_jzazbz(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to the Jzazbz color space."""
    func = _get_backend_func("xyz_to_jzazbz")
    return np.asarray(func(xyz))


def jzazbz_to_xyz_d65(jzazbz: np.ndarray) -> np.ndarray:
    """Converts the Jzazbz color space to CIE XYZ (D65)."""
    func = _get_backend_func("jzazbz_to_xyz")
    return np.asarray(func(jzazbz))
