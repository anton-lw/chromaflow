# src/chromaflow/backends/jax_backend.py
import jax
import jax.numpy as jnp
from jax import jit
from ..core import constants as const

FloatArray = jax.Array

@jit
def srgb_oetf(linear_rgb: FloatArray) -> FloatArray:
    return jnp.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * (jnp.power(linear_rgb, 1.0 / 2.4)) - 0.055,
    )

@jit
def srgb_eotf(srgb: FloatArray) -> FloatArray:
    return jnp.where(
        srgb <= 0.04045,
        srgb / 12.92,
        jnp.power((srgb + 0.055) / 1.055, 2.4),
    )

@jit
def xyz_to_lab(xyz: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    xyz_ref = xyz / whitepoint_xyz
    def f(t: FloatArray) -> FloatArray:
        return jnp.where(t > const.CIE_E, jnp.cbrt(t), (const.CIE_K * t + 16) / 116)
    fx, fy, fz = f(xyz_ref[..., 0]), f(xyz_ref[..., 1]), f(xyz_ref[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return jnp.stack([L, a, b], axis=-1)

@jit
def lab_to_xyz(lab: FloatArray, whitepoint_xyz: FloatArray) -> FloatArray:
    """Converts CIELAB to CIE XYZ."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t: FloatArray) -> FloatArray:
        t_3 = jnp.power(t, 3)
        return jnp.where(t_3 > const.CIE_E, t_3, (116 * t - 16) / const.CIE_K)

    xyz_ref = jnp.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1)
    return xyz_ref * whitepoint_xyz

@jit
def rgb_to_xyz(rgb_linear: FloatArray, matrix: FloatArray) -> FloatArray:
    return jnp.dot(rgb_linear, matrix.T)

@jit
def xyz_to_rgb(xyz: FloatArray, matrix_inv: FloatArray) -> FloatArray:
    return jnp.dot(xyz, matrix_inv.T)

@jit
def xyz_to_oklab(xyz: FloatArray) -> FloatArray:
    lms = jnp.dot(xyz, const.OKLAB_M1.T)
    lms_prime = jnp.cbrt(lms)
    return jnp.dot(lms_prime, const.OKLAB_M2.T)

@jit
def oklab_to_xyz(oklab: FloatArray) -> FloatArray:
    lms_prime = jnp.dot(oklab, const.OKLAB_M2_INV.T)
    lms = jnp.power(lms_prime, 3)
    return jnp.dot(lms, const.OKLAB_M1_INV.T)

@jit
def oklab_to_oklch(oklab: FloatArray) -> FloatArray:
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    C = jnp.sqrt(a**2 + b**2)
    h = jnp.degrees(jnp.arctan2(b, a))
    h = jnp.where(h < 0, h + 360, h)
    return jnp.stack([L, C, h], axis=-1)

@jit
def oklch_to_oklab(oklch: FloatArray) -> FloatArray:
    L, C, h = oklch[..., 0], oklch[..., 1], oklch[..., 2]
    h_rad = jnp.radians(h)
    a = C * jnp.cos(h_rad)
    b = C * jnp.sin(h_rad)
    return jnp.stack([L, a, b], axis=-1)

@jit
def xyz_to_jzazbz(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Jzazbz."""
    lms = jnp.dot(xyz, const.M_XYZ_TO_HPE.T)
    lms_p = jnp.cbrt(lms)
    
    Iz = const.JZAZBZ_B * lms_p[..., 0] + const.JZAZBZ_G * lms_p[..., 1]
    az = const.JZAZBZ_C1 * lms_p[..., 0] + const.JZAZBZ_C2 * lms_p[..., 1] + const.JZAZBZ_C3 * lms_p[..., 2]
    bz = lms_p[..., 0] - lms_p[..., 1]
    
    Jz = (Iz - const.JZAZBZ_I_CONST) * 0.5
    
    return jnp.stack([Jz, az, bz], axis=-1)

@jit
def jzazbz_to_xyz(jzazbz: FloatArray) -> FloatArray:
    """Converts Jzazbz to CIE XYZ D65."""
    Jz, az, bz = jzazbz[..., 0], jzazbz[..., 1], jzazbz[..., 2]
    
    Iz = 2 * Jz + const.JZAZBZ_I_CONST
    
    m_inv = jnp.linalg.inv(jnp.array([
        [const.JZAZBZ_B, const.JZAZBZ_G, 0],
        [const.JZAZBZ_C1, const.JZAZBZ_C2, const.JZAZBZ_C3],
        [1, -1, 0]
    ]))
    
    iz_az_bz = jnp.stack([Iz, az, bz], axis=-1)
    lms_p = jnp.dot(iz_az_bz, m_inv.T)
    
    lms = jnp.power(lms_p, 3)
    
    xyz = jnp.dot(lms, const.M_HPE_TO_XYZ.T)
    
    return xyz