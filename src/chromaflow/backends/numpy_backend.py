# src/chromaflow/backends/numpy_backend.py
import numpy as np
from numpy.typing import NDArray

from ..core import constants as const

FloatArray = NDArray[np.float64]


def srgb_oetf(linear_rgb: FloatArray) -> FloatArray:
    result = np.empty_like(linear_rgb)
    mask = linear_rgb <= 0.0031308
    result[mask] = linear_rgb[mask] * 12.92
    result[~mask] = 1.055 * np.power(linear_rgb[~mask], 1.0 / 2.4) - 0.055
    return result


def srgb_eotf(srgb: FloatArray) -> FloatArray:
    result = np.empty_like(srgb)
    mask = srgb <= 0.04045
    result[mask] = srgb[mask] / 12.92
    result[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)
    return result


def xyz_to_lab(xyz: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    xyz_ref = xyz / whitepoint_xyz

    def f(t: FloatArray) -> FloatArray:
        return np.where(t > const.CIE_E, np.cbrt(t), (const.CIE_K * t + 16) / 116)

    fx, fy, fz = f(xyz_ref[..., 0]), f(xyz_ref[..., 1]), f(xyz_ref[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t: FloatArray) -> FloatArray:
        return np.where(t**3 > const.CIE_E, t**3, (116 * t - 16) / const.CIE_K)

    xyz_ref = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1)
    return xyz_ref * whitepoint_xyz


def rgb_to_xyz(rgb_linear: FloatArray, matrix: FloatArray) -> FloatArray:
    return np.asarray(np.dot(rgb_linear, matrix.T))


def xyz_to_rgb(xyz: FloatArray, matrix_inv: FloatArray) -> FloatArray:
    return np.asarray(np.dot(xyz, matrix_inv.T))


# --- Oklab / Oklch Functions ---
def xyz_to_oklab(xyz: FloatArray) -> FloatArray:
    lms = np.dot(xyz, const.OKLAB_M1.T)
    lms_prime = np.cbrt(lms)
    return np.asarray(np.dot(lms_prime, const.OKLAB_M2.T))


def oklab_to_xyz(oklab: FloatArray) -> FloatArray:
    lms_prime = np.dot(oklab, const.OKLAB_M2_INV.T)
    lms = np.power(lms_prime, 3)
    return np.asarray(np.dot(lms, const.OKLAB_M1_INV.T))


def oklab_to_oklch(oklab: FloatArray) -> FloatArray:
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a))
    h = np.where(h < 0, h + 360, h)
    return np.stack([L, C, h], axis=-1)


def oklch_to_oklab(oklch: FloatArray) -> FloatArray:
    L, C, h = oklch[..., 0], oklch[..., 1], oklch[..., 2]
    h_rad = np.radians(h)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    return np.stack([L, a, b], axis=-1)


def _st2084_inverse_eotf(values: FloatArray) -> FloatArray:
    values = np.clip(values, 0.0, None)
    powered = np.power(values / 10000.0, const.JZAZBZ_M1)
    numerator = const.JZAZBZ_C1 + const.JZAZBZ_C2 * powered
    denominator = 1.0 + const.JZAZBZ_C3 * powered
    return np.power(numerator / denominator, const.JZAZBZ_M2)


def _st2084_eotf(values: FloatArray) -> FloatArray:
    values = np.clip(values, 0.0, None)
    powered = np.power(values, 1.0 / const.JZAZBZ_M2)
    numerator = np.maximum(powered - const.JZAZBZ_C1, 0.0)
    denominator = const.JZAZBZ_C2 - const.JZAZBZ_C3 * powered
    return np.power(numerator / denominator, 1.0 / const.JZAZBZ_M1) * 10000.0


def xyz_to_jzazbz(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Jzazbz."""
    x_p = const.JZAZBZ_B * xyz[..., 0] - (const.JZAZBZ_B - 1.0) * xyz[..., 2]
    y_p = const.JZAZBZ_G * xyz[..., 1] - (const.JZAZBZ_G - 1.0) * xyz[..., 0]
    xyz_p = np.stack([x_p, y_p, xyz[..., 2]], axis=-1)
    lms = np.dot(xyz_p, const.JZAZBZ_XYZ_TO_LMS_MATRIX.T)
    lms_p = _st2084_inverse_eotf(lms)
    izazbz = np.dot(lms_p, const.JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX.T)
    iz = izazbz[..., 0]
    jz = ((1.0 + const.JZAZBZ_D) * iz) / (1.0 + const.JZAZBZ_D * iz) - const.JZAZBZ_D0
    return np.stack([jz, izazbz[..., 1], izazbz[..., 2]], axis=-1)


def jzazbz_to_xyz(jzazbz: FloatArray) -> FloatArray:
    """Converts Jzazbz to CIE XYZ D65."""
    jz, az, bz = jzazbz[..., 0], jzazbz[..., 1], jzazbz[..., 2]
    iz = (jz + const.JZAZBZ_D0) / (
        1.0 + const.JZAZBZ_D - const.JZAZBZ_D * (jz + const.JZAZBZ_D0)
    )
    izazbz = np.stack([iz, az, bz], axis=-1)
    lms_p = np.dot(izazbz, const.JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX.T)
    lms = _st2084_eotf(lms_p)
    xyz_p = np.dot(lms, const.JZAZBZ_LMS_TO_XYZ_MATRIX.T)
    x = (xyz_p[..., 0] + (const.JZAZBZ_B - 1.0) * xyz_p[..., 2]) / const.JZAZBZ_B
    y = (xyz_p[..., 1] + (const.JZAZBZ_G - 1.0) * x) / const.JZAZBZ_G
    return np.stack([x, y, xyz_p[..., 2]], axis=-1)
