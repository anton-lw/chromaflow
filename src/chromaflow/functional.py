# src/chromaflow/functional.py
import numpy as np
from typing import Any

from . import config
from .backends import numpy_backend, jax_backend, numba_backend
from .core import constants as const
from .core.constants import (
    ILLUMINANT_D65_XYZ,
    SRGB_TO_XYZ_MATRIX,
    XYZ_TO_SRGB_MATRIX,
    P3_TO_XYZ_MATRIX,
    XYZ_TO_P3_MATRIX,
)

_DISPATCH = {
    "numpy": numpy_backend,
    "jax": jax_backend,
    "numba": numba_backend,
}

def _get_backend_func(name: str) -> Any:
    """Dynamically get a function from the currently configured backend."""
    backend_name = config.get_backend()
    try:
        backend_module = _DISPATCH[backend_name]
    except KeyError:
        raise config.BackendConfigurationError(f"Backend '{backend_name}' not found or installed.") from None
    
    func = getattr(backend_module, name, None)
    if func is None:
        # Fallback to numpy if the function is not implemented in the selected backend
        func = getattr(numpy_backend, name, None)
        if func is None:
            raise NotImplementedError(f"Function '{name}' not implemented for backend '{backend_name}' or numpy fallback.")
    return func


def srgb_to_srgb_linear(srgb: np.ndarray) -> np.ndarray:
    """Converts non-linear sRGB to linear sRGB."""
    func = _get_backend_func('srgb_eotf')
    return np.asarray(func(srgb))

def srgb_linear_to_srgb(srgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear sRGB to non-linear sRGB."""
    func = _get_backend_func('srgb_oetf')
    return np.asarray(func(srgb_linear))


def p3_d65_to_p3_d65_linear(p3: np.ndarray) -> np.ndarray:
    """Converts non-linear Display P3 to linear Display P3."""
    # Display P3 uses the same transfer function as sRGB.
    return srgb_to_srgb_linear(p3)

def p3_d65_linear_to_p3_d65(p3_linear: np.ndarray) -> np.ndarray:
    """Converts linear Display P3 to non-linear Display P3."""
    # Display P3 uses the same transfer function as sRGB.
    return srgb_linear_to_srgb(p3_linear)

def p3_d65_linear_to_xyz_d65(p3_linear: np.ndarray) -> np.ndarray:
    """Converts linear Display P3 to CIE XYZ (D65)."""
    func = _get_backend_func('rgb_to_xyz')
    return np.asarray(func(p3_linear, P3_TO_XYZ_MATRIX))

def xyz_d65_to_p3_d65_linear(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to linear Display P3."""
    func = _get_backend_func('xyz_to_rgb')
    return np.asarray(func(xyz, XYZ_TO_P3_MATRIX))

def srgb_linear_to_xyz_d65(srgb_linear: np.ndarray) -> np.ndarray:
    """Converts linear sRGB to CIE XYZ (D65)."""
    func = _get_backend_func('rgb_to_xyz')
    return np.asarray(func(srgb_linear, SRGB_TO_XYZ_MATRIX))

def xyz_d65_to_srgb_linear(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to linear sRGB."""
    func = _get_backend_func('xyz_to_rgb')
    return np.asarray(func(xyz, XYZ_TO_SRGB_MATRIX))


def xyz_d65_to_lab_d65(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ to CIELAB, assuming a D65 illuminant."""
    func = _get_backend_func('xyz_to_lab')
    return np.asarray(func(xyz, ILLUMINANT_D65_XYZ))

def lab_d65_to_xyz_d65(lab: np.ndarray) -> np.ndarray:
    """Converts CIELAB to CIE XYZ, assuming a D65 illuminant."""
    func = _get_backend_func('lab_to_xyz')
    return np.asarray(func(lab, ILLUMINANT_D65_XYZ))

def xyz_d65_to_oklab(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to Oklab."""
    func = _get_backend_func('xyz_to_oklab')
    return np.asarray(func(xyz))

def oklab_to_xyz_d65(oklab: np.ndarray) -> np.ndarray:
    """Converts Oklab to CIE XYZ (D65)."""
    func = _get_backend_func('oklab_to_xyz')
    return np.asarray(func(oklab))

def oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    """Converts Oklab to Oklch (cylindrical coordinates)."""
    func = _get_backend_func('oklab_to_oklch')
    return np.asarray(func(oklab))

def oklch_to_oklab(oklch: np.ndarray) -> np.ndarray:
    """Converts Oklch to Oklab (cartesian coordinates)."""
    func = _get_backend_func('oklch_to_oklab')
    return np.asarray(func(oklch))

def xyz_d65_to_jzazbz(xyz: np.ndarray) -> np.ndarray:
    """Converts CIE XYZ (D65) to the Jzazbz color space."""
    func = _get_backend_func('xyz_to_jzazbz')
    return np.asarray(func(xyz))

def jzazbz_to_xyz_d65(jzazbz: np.ndarray) -> np.ndarray:
    """Converts the Jzazbz color space to CIE XYZ (D65)."""
    func = _get_backend_func('jzazbz_to_xyz')
    return np.asarray(func(jzazbz))