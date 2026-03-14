# src/chromaflow/backends/jax_backend.py
import jax
import jax.numpy as jnp
from jax import jit

from ..core import constants as const

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

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
def _st2084_inverse_eotf(values: FloatArray) -> FloatArray:
    values = jnp.clip(values, 0.0, None)
    powered = jnp.power(values / 10000.0, const.JZAZBZ_M1)
    numerator = const.JZAZBZ_C1 + const.JZAZBZ_C2 * powered
    denominator = 1.0 + const.JZAZBZ_C3 * powered
    return jnp.power(numerator / denominator, const.JZAZBZ_M2)


@jit
def _st2084_eotf(values: FloatArray) -> FloatArray:
    values = jnp.clip(values, 0.0, None)
    powered = jnp.power(values, 1.0 / const.JZAZBZ_M2)
    numerator = jnp.maximum(powered - const.JZAZBZ_C1, 0.0)
    denominator = const.JZAZBZ_C2 - const.JZAZBZ_C3 * powered
    return jnp.power(numerator / denominator, 1.0 / const.JZAZBZ_M1) * 10000.0


@jit
def xyz_to_jzazbz(xyz: FloatArray) -> FloatArray:
    """Converts CIE XYZ D65 to Jzazbz."""
    x_p = const.JZAZBZ_B * xyz[..., 0] - (const.JZAZBZ_B - 1.0) * xyz[..., 2]
    y_p = const.JZAZBZ_G * xyz[..., 1] - (const.JZAZBZ_G - 1.0) * xyz[..., 0]
    xyz_p = jnp.stack([x_p, y_p, xyz[..., 2]], axis=-1)
    lms = jnp.dot(xyz_p, const.JZAZBZ_XYZ_TO_LMS_MATRIX.T)
    lms_p = _st2084_inverse_eotf(lms)
    izazbz = jnp.dot(lms_p, const.JZAZBZ_LMS_P_TO_IZAZBZ_MATRIX.T)
    iz = izazbz[..., 0]
    jz = ((1.0 + const.JZAZBZ_D) * iz) / (1.0 + const.JZAZBZ_D * iz) - const.JZAZBZ_D0
    return jnp.stack([jz, izazbz[..., 1], izazbz[..., 2]], axis=-1)


@jit
def jzazbz_to_xyz(jzazbz: FloatArray) -> FloatArray:
    """Converts Jzazbz to CIE XYZ D65."""
    jz, az, bz = jzazbz[..., 0], jzazbz[..., 1], jzazbz[..., 2]
    iz = (jz + const.JZAZBZ_D0) / (
        1.0 + const.JZAZBZ_D - const.JZAZBZ_D * (jz + const.JZAZBZ_D0)
    )
    izazbz = jnp.stack([iz, az, bz], axis=-1)
    lms_p = jnp.dot(izazbz, const.JZAZBZ_IZAZBZ_TO_LMS_P_MATRIX.T)
    lms = _st2084_eotf(lms_p)
    xyz_p = jnp.dot(lms, const.JZAZBZ_LMS_TO_XYZ_MATRIX.T)
    x = (xyz_p[..., 0] + (const.JZAZBZ_B - 1.0) * xyz_p[..., 2]) / const.JZAZBZ_B
    y = (xyz_p[..., 1] + (const.JZAZBZ_G - 1.0) * x) / const.JZAZBZ_G
    return jnp.stack([x, y, xyz_p[..., 2]], axis=-1)
