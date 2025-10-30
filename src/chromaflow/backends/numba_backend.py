# src/chromaflow/backends/numba_backend.py
import numpy as np
import numba
from numpy.typing import NDArray
from ..core import constants as const

FloatArray = NDArray[np.float64]

@numba.njit(fastmath=True)
def _power(x: float, p: float) -> float:
    """A numba-jittable power function."""
    return x ** p

@numba.njit(fastmath=True, parallel=True)
def srgb_oetf(linear_rgb: FloatArray) -> FloatArray:
    """Opto-electrical transfer function for sRGB (linear to non-linear)."""
    # Using parallel=True and prange for element-wise operations is very efficient.
    result = np.empty_like(linear_rgb)
    for i in numba.prange(linear_rgb.size):
        val = linear_rgb.flat[i]
        if val <= 0.0031308:
            result.flat[i] = val * 12.92
        else:
            result.flat[i] = 1.055 * (_power(val, 1.0 / 2.4)) - 0.055
    return result

@numba.njit(fastmath=True, parallel=True)
def srgb_eotf(srgb: FloatArray) -> FloatArray:
    """Electro-optical transfer function for sRGB (non-linear to linear)."""
    result = np.empty_like(srgb)
    for i in numba.prange(srgb.size):
        val = srgb.flat[i]
        if val <= 0.04045:
            result.flat[i] = val / 12.92
        else:
            result.flat[i] = _power((val + 0.055) / 1.055, 2.4)
    return result

@numba.njit(fastmath=True, parallel=True)
def xyz_to_lab(xyz: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ to CIELAB."""
    result = np.empty_like(xyz)
    for i in numba.prange(xyz.shape[0]):
        xyz_ref_x = xyz[i, 0] / whitepoint_xyz[0]
        xyz_ref_y = xyz[i, 1] / whitepoint_xyz[1]
        xyz_ref_z = xyz[i, 2] / whitepoint_xyz[2]

        # f function logic
        fx = _power(xyz_ref_x, 1/3) if xyz_ref_x > const.CIE_E else (const.CIE_K * xyz_ref_x + 16) / 116
        fy = _power(xyz_ref_y, 1/3) if xyz_ref_y > const.CIE_E else (const.CIE_K * xyz_ref_y + 16) / 116
        fz = _power(xyz_ref_z, 1/3) if xyz_ref_z > const.CIE_E else (const.CIE_K * xyz_ref_z + 16) / 116
        
        result[i, 0] = 116 * fy - 16
        result[i, 1] = 500 * (fx - fy)
        result[i, 2] = 200 * (fy - fz)
    return result

@numba.njit(fastmath=True, parallel=True)
def lab_to_xyz(lab: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    """Converts CIELAB to CIE XYZ."""
    result = np.empty_like(lab)
    for i in numba.prange(lab.shape[0]):
        L, a, b = lab[i, 0], lab[i, 1], lab[i, 2]
        
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        # f_inv function logic, applied to scalars for Numba efficiency
        fx_3 = fx**3
        fy_3 = fy**3
        fz_3 = fz**3
        
        xr = fx_3 if fx_3 > const.CIE_E else (116 * fx - 16) / const.CIE_K
        yr = fy_3 if fy_3 > const.CIE_E else (116 * fy - 16) / const.CIE_K
        zr = fz_3 if fz_3 > const.CIE_E else (116 * fz - 16) / const.CIE_K
        
        result[i, 0] = xr * whitepoint_xyz[0]
        result[i, 1] = yr * whitepoint_xyz[1]
        result[i, 2] = zr * whitepoint_xyz[2]
    return result

@numba.njit(fastmath=True)
def rgb_to_xyz(rgb_linear: FloatArray, matrix: FloatArray) -> FloatArray:
    """Converts linear RGB to CIE XYZ using a given matrix."""
    # np.dot is highly optimized and Numba compiles it efficiently.
    return np.dot(rgb_linear, matrix.T)

@numba.njit(fastmath=True)
def xyz_to_rgb(xyz: FloatArray, matrix_inv: FloatArray) -> FloatArray:
    """Converts CIE XYZ to linear RGB using a given inverse matrix."""
    return np.dot(xyz, matrix_inv.T)

# --- Oklab / Oklch Functions ---
@numba.njit(fastmath=True)
def xyz_to_oklab(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Oklab."""
    lms = np.dot(xyz, const.OKLAB_M1.T)
    lms_prime = np.cbrt(lms)
    return np.dot(lms_prime, const.OKLAB_M2.T)

@numba.njit(fastmath=True)
def oklab_to_xyz(oklab: FloatArray) -> FloatArray:
    """Converts Oklab to CIE XYZ D65."""
    lms_prime = np.dot(oklab, const.OKLAB_M2_INV.T)
    lms = _power(lms_prime, 3)
    return np.dot(lms, const.OKLAB_M1_INV.T)

@numba.njit(fastmath=True, parallel=True)
def oklab_to_oklch(oklab: FloatArray) -> FloatArray:
    """Converts Oklab to Oklch."""
    result = np.empty_like(oklab)
    for i in numba.prange(oklab.shape[0]):
        L, a, b = oklab[i, 0], oklab[i, 1], oklab[i, 2]
        C = np.sqrt(a**2 + b**2)
        h = np.degrees(np.arctan2(b, a))
        result[i, 0] = L
        result[i, 1] = C
        result[i, 2] = h if h >= 0 else h + 360
    return result

@numba.njit(fastmath=True, parallel=True)
def oklch_to_oklab(oklch: FloatArray) -> FloatArray:
    """Converts Oklch to Oklab."""
    result = np.empty_like(oklch)
    for i in numba.prange(oklch.shape[0]):
        L, C, h = oklch[i, 0], oklch[i, 1], oklch[i, 2]
        h_rad = np.radians(h)
        a = C * np.cos(h_rad)
        b = C * np.sin(h_rad)
        result[i, 0] = L
        result[i, 1] = a
        result[i, 2] = b
    return result

@numba.njit(fastmath=True, parallel=True)
def xyz_to_jzazbz(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Jzazbz."""
    lms = np.dot(xyz, const.M_XYZ_TO_HPE.T)
    lms_p = np.cbrt(lms)
    
    result = np.empty_like(xyz)
    for i in numba.prange(xyz.shape[0]):
        Iz = const.JZAZBZ_B * lms_p[i, 0] + const.JZAZBZ_G * lms_p[i, 1]
        az = const.JZAZBZ_C1 * lms_p[i, 0] + const.JZAZBZ_C2 * lms_p[i, 1] + const.JZAZBZ_C3 * lms_p[i, 2]
        bz = lms_p[i, 0] - lms_p[i, 1]
        
        Jz = (Iz - const.JZAZBZ_I_CONST) * 0.5
        
        result[i, 0] = Jz
        result[i, 1] = az
        result[i, 2] = bz
    return result

@numba.njit(fastmath=True)
def jzazbz_to_xyz(jzazbz: FloatArray) -> FloatArray:
    """Converts Jzazbz to CIE XYZ D65."""
    m_inv = np.linalg.inv(np.array([
        [const.JZAZBZ_B, const.JZAZBZ_G, 0.0],
        [const.JZAZBZ_C1, const.JZAZBZ_C2, const.JZAZBZ_C3],
        [1.0, -1.0, 0.0]
    ]))

    iz_az_bz = np.empty_like(jzazbz)
    for i in range(jzazbz.shape[0]):
        Jz, az, bz = jzazbz[i, 0], jzazbz[i, 1], jzazbz[i, 2]
        Iz = 2 * Jz + const.JZAZBZ_I_CONST
        iz_az_bz[i, 0] = Iz
        iz_az_bz[i, 1] = az
        iz_az_bz[i, 2] = bz
        
    lms_p = np.dot(iz_az_bz, m_inv.T)
    lms = np.power(lms_p, 3)
    xyz = np.dot(lms, const.M_HPE_TO_XYZ.T)
    
    return xyz