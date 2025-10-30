# src/chromaflow/backends/numpy_backend.py
import numpy as np
from numpy.typing import NDArray
from ..core import constants as const

FloatArray = NDArray[np.float64]

def srgb_oetf(linear_rgb: FloatArray) -> FloatArray:
    return np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * (np.power(linear_rgb, 1.0 / 2.4)) - 0.055,
    )

def srgb_eotf(srgb: FloatArray) -> FloatArray:
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )

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
    return np.dot(rgb_linear, matrix.T)

def xyz_to_rgb(xyz: FloatArray, matrix_inv: FloatArray) -> FloatArray:
    return np.dot(xyz, matrix_inv.T)

# --- Oklab / Oklch Functions ---
def xyz_to_oklab(xyz: FloatArray) -> FloatArray:
    lms = np.dot(xyz, const.OKLAB_M1.T)
    lms_prime = np.cbrt(lms)
    return np.dot(lms_prime, const.OKLAB_M2.T)

def oklab_to_xyz(oklab: FloatArray) -> FloatArray:
    lms_prime = np.dot(oklab, const.OKLAB_M2_INV.T)
    lms = np.power(lms_prime, 3)
    return np.dot(lms, const.OKLAB_M1_INV.T)

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

def xyz_to_jzazbz(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Jzazbz."""
    # Step 1: XYZ to LMS cone space (Hunt-Pointer-Estevez)
    lms = np.dot(xyz, const.M_XYZ_TO_HPE.T)

    # Non-linear compression (perceptual quantization)
    lms_p = np.cbrt(lms)
    
    # LMS' to Izazbz
    Iz = const.JZAZBZ_B * lms_p[..., 0] + const.JZAZBZ_G * lms_p[..., 1]
    az = const.JZAZBZ_C1 * lms_p[..., 0] + const.JZAZBZ_C2 * lms_p[..., 1] + const.JZAZBZ_C3 * lms_p[..., 2]
    bz = lms_p[..., 0] - lms_p[..., 1]

    # Calculate Jz
    Jz = (Iz - const.JZAZBZ_I_CONST) * 0.5

    return np.stack([Jz, az, bz], axis=-1)

def jzazbz_to_xyz(jzazbz: FloatArray) -> FloatArray:
    """Converts Jzazbz to CIE XYZ D65."""
    Jz, az, bz = jzazbz[..., 0], jzazbz[..., 1], jzazbz[..., 2]

    # Jz to Iz
    Iz = 2 * Jz + const.JZAZBZ_I_CONST
    
    # SIzazbz to LMS'
    # The matrix for this is derived from the forward equations.
    m_inv = np.linalg.inv(np.array([
        [const.JZAZBZ_B, const.JZAZBZ_G, 0],
        [const.JZAZBZ_C1, const.JZAZBZ_C2, const.JZAZBZ_C3],
        [1, -1, 0]
    ]))
    
    iz_az_bz = np.stack([Iz, az, bz], axis=-1)
    lms_p = np.dot(iz_az_bz, m_inv.T)

    # Inverse non-linear compression
    lms = np.power(lms_p, 3)

    # LMS to XYZ
    xyz = np.dot(lms, const.M_HPE_TO_XYZ.T)
    
    return xyz